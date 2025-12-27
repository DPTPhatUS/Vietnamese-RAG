from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from neo4j import GraphDatabase

from vietrag.config import Neo4jConfig
from vietrag.types import RetrievalDocument
from vietrag.llm.qwen import QwenClient


FEW_SHOT_EXAMPLES = [
    {
        "question": "Phương pháp điều trị cho bệnh [U lympho sau phúc mạc] là gì?",
        "query": """
MATCH (d:`ĐIỀU TRỊ`)
WHERE d.tên_bệnh = 'U lympho sau phúc mạc'
RETURN {
  bệnh: d.tên_bệnh,
  phương_pháp: d.phương_pháp
} AS điều_trị
"""
    },
    {
        "question": "Nguyên nhân của bệnh [Chảy máu khoảng cách sau phúc mạc] là gì?",
        "query": """
MATCH (b:`BỆNH`)
WHERE b.tên_bệnh = 'Chảy máu khoảng cách sau phúc mạc'
RETURN {
  bệnh: b.tên_bệnh,
  nguyên_nhân: b.nguyên_nhân
} AS nguyên_nhân
"""
    },
    {
        "question": "Triệu chứng của bệnh [Chảy máu khoảng cách sau phúc mạc] là gì?",
        "query": """
MATCH (s:`TRIỆU CHỨNG`)
WHERE s.tên_bệnh = 'Chảy máu khoảng cách sau phúc mạc'
RETURN {
  bệnh: s.tên_bệnh,
  triệu_chứng: s.triệu_chứng
} AS triệu_chứng
"""
    },
    {
        "question": "Những bệnh lý nào có thể xuất hiện khi có triệu chứng [Khóc và đau]?",
        "query": """
MATCH (s:`TRIỆU CHỨNG`)
WHERE s.triệu_chứng CONTAINS 'Khóc và đau'
RETURN {
  bệnh: s.tên_bệnh,
  triệu_chứng: s.triệu_chứng
} AS triệu_chứng
"""
    },
    {
        "question": "Có những loại thuốc phổ biến nào để điều trị bệnh [Chảy máu khoảng cách sau phúc mạc]?",
        "query": """
MATCH (m:`THUỐC`)
WHERE m.tên_bệnh = 'Chảy máu khoảng cách sau phúc mạc'
RETURN {
  bệnh: m.tên_bệnh,
  thuốc_phổ_biến: m.thuốc_phổ_biến
} AS thuốc
"""
    },
]

# Took this directly from Neo4j
DEFAULT_SCHEMA_DESCRIPTION = """
`BỆNH`: loại_bệnh, mô_tả_bệnh, nguyên_nhân, tên_bệnh
`LỜI KHUYÊN`: cách_phòng_tránh, không_nên_ăn_thực_phẩm_chứa, nên_ăn_thực_phẩm_chứa, tên_bệnh, đề_xuất_món_ăn
`THUỐC`: thuốc_phổ_biến, thông_tin_thuốc, tên_bệnh, đề_xuất_thuốc
`TRIỆU CHỨNG`: kiểm_tra, triệu_chứng, tên_bệnh, đối_tượng_dễ_mắc_bệnh
`ĐIỀU TRỊ`: khoa_điều_trị, phương_pháp, tên_bệnh, tỉ_lệ_chữa_khỏi
""".strip()


logger = logging.getLogger(__name__)


class VietMedKGRetriever:
    def __init__(self, config: Neo4jConfig, llm: Optional["QwenClient"] = None):
        self.config = config
        self.llm = llm
        self._driver = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
        self.system_prompt = (
            "Bạn là chuyên gia Neo4j cho đồ thị Y học Cổ truyền Việt Nam. "
            "Hãy tạo truy vấn Cypher chính xác và chỉ trả về câu lệnh."
        )
        self._schema_description: Optional[str] = None

    def close(self) -> None:
        self._driver.close()

    def search(self, query: str) -> List[RetrievalDocument]:
        if not self.llm:
            logger.warning("KG retriever requires an LLM for Cypher generation; returning empty result")
            return []
        documents = self._cypher_workflow(query)
        return documents

    def _cypher_workflow(self, question: str) -> List[RetrievalDocument]:
        cypher = self._generate_cypher(question)
        # print("Cypher: ", cypher)
        if not cypher:
            return []
        try:
            records = self._run_cypher(cypher)
            # print("Records: ", len(records))
        except Exception as exc:
            logger.warning("Cypher execution failed: %s", exc)
            return []
        if not records:
            return []
        return self._records_to_documents(records[:256], cypher)

    def _generate_cypher(self, question: str) -> Optional[str]:
        if not self.llm:
            return None
        schema = self._get_schema_description()
        examples = "\n\n".join(
            f"Ví dụ hỏi: {ex['question']}\nCypher: {ex['query']}" for ex in FEW_SHOT_EXAMPLES
        )
        user_prompt = (
            f"Sơ đồ dữ liệu:\n{schema}\n\n"
            f"{examples}\n\n"
            f"Hãy viết truy vấn Cypher duy nhất để trả lời câu hỏi sau. "
            f"Chỉ dùng thuộc tính có trong sơ đồ dữ liệu.\n"
            f"Câu hỏi: {question}\n"
            "Cypher:"
        )
        try:
            response = self.llm.generate(self.system_prompt, user_prompt, temperature=0.05, top_p=1.0, top_k=0, min_p=0.0)
        except Exception as exc:
            logger.warning("Failed to generate Cypher query: %s", exc)
            return None
        return self._extract_cypher(response)

    def _get_schema_description(self) -> str:
        if self._schema_description:
            return self._schema_description
        query = """
        CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName
        WITH nodeType, collect(DISTINCT propertyName) AS props
        RETURN nodeType, props
        ORDER BY nodeType
        """
        try:
            with self._driver.session(database=self.config.database) as session:
                rows = session.run(query)
                sections = []
                for row in rows:
                    node_type = row["nodeType"]
                    if isinstance(node_type, list):
                        label = ":".join(label.strip(":") for label in node_type if label)
                    else:
                        label = str(node_type).strip(":")
                    props = ", ".join(sorted(row["props"]))
                    sections.append(f"{label or 'NODE'}: {props}")
                if sections:
                    self._schema_description = "\n".join(sections)
                else:
                    self._schema_description = DEFAULT_SCHEMA_DESCRIPTION
        except Exception:
            self._schema_description = DEFAULT_SCHEMA_DESCRIPTION
        return self._schema_description

    def _extract_cypher(self, raw: str) -> Optional[str]:
        if not raw:
            return None
        if "```" in raw:
            blocks = re.findall(r"```(?:cypher)?\s*([\s\S]+?)```", raw, flags=re.IGNORECASE)
            if blocks:
                return blocks[0].strip()
        match = re.search(r"(?is)(match .*?)$", raw.strip())
        if match:
            return match.group(1).strip()
        return raw.strip()

    def _run_cypher(self, cypher: str):
        with self._driver.session(database=self.config.database) as session:
            return list(session.run(cypher))

    def _records_to_documents(
        self,
        records,
        cypher: str,
    ) -> List[RetrievalDocument]:
        documents: List[RetrievalDocument] = []
        for idx, record in enumerate(records):
            document = self._record_to_document(record, cypher, idx)
            if document:
                documents.append(document)
        return documents

    def _record_to_document(self, record, cypher: str, rank: int) -> Optional[RetrievalDocument]:
        metadata: Dict[str, str] = {
            "source": "knowledge_graph",
            "cypher_query": cypher,
            "rank": str(rank + 1),
        }
        score = 1.0
        text = json.dumps(record.data(), ensure_ascii=False, indent=2)
        return RetrievalDocument(text=text, score=score, metadata=metadata)


__all__ = ["VietMedKGRetriever"]
