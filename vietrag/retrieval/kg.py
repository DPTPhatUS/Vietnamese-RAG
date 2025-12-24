from __future__ import annotations

from typing import Dict, List, Tuple

from neo4j import GraphDatabase

from vietrag.config import Neo4jConfig
from vietrag.types import RetrievalDocument


TEXT_PRIORITY = [
	"summary",
	"description",
	"remedy",
	"notes",
	"mô_tả_bệnh",
	"triệu_chứng",
	"thông_tin_thuốc",
	"đề_xuất_thuốc",
	"phương_pháp",
	"nên_ăn_thực_phẩm_chứa",
	"không_nên_ăn_thực_phẩm_chứa",
	"cách_phòng_tránh",
	"thuốc_phổ_biến",
	"đối_tượng_dễ_mắc_bệnh",
]


VN_KEY_ALIASES = {
	"tên_bệnh": "name",
	"mô_tả_bệnh": "description_vi",
	"triệu_chứng": "symptom_vi",
	"phương_pháp": "method_vi",
	"thông_tin_thuốc": "medication_vi",
}


class VietMedKGRetriever:
	def __init__(self, config: Neo4jConfig):
		self.config = config
		self._driver = GraphDatabase.driver(
			config.uri,
			auth=(config.username, config.password),
		)

	def close(self) -> None:
		self._driver.close()

	def _format_node(self, node) -> Tuple[Dict[str, str], str]:
		props = {k: str(v) for k, v in dict(node).items()}
		labels = list(node.labels)
		props.setdefault("labels", ",".join(labels))
		if "name" not in props and "tên_bệnh" in props:
			props["name"] = props["tên_bệnh"]
		for vn_key, alias in VN_KEY_ALIASES.items():
			if vn_key in props and alias not in props:
				props[alias] = props[vn_key]
		text_parts: List[str] = []
		seen: set[str] = set()
		for key in TEXT_PRIORITY:
			value = props.get(key)
			if not value:
				continue
			cleaned = str(value).strip()
			if cleaned and cleaned not in seen:
				text_parts.append(cleaned)
				seen.add(cleaned)
		text = "\n".join(text_parts)
		props.setdefault("summary", text or props.get("name", ""))
		return props, text

	def search(self, query: str, top_k: int = 5) -> List[RetrievalDocument]:
		cypher = """
		CALL db.index.fulltext.queryNodes($index, $q)
		YIELD node, score
		RETURN node, score
		ORDER BY score DESC
		LIMIT $k
		"""
		params = {"index": self.config.fulltext_index, "q": query, "k": top_k}
		records = []
		try:
			with self._driver.session(database=self.config.database) as session:
				result = session.run(cypher, **params)
				records = list(result)
		except Exception:
			fallback = """
			MATCH (n)
			WHERE any(key IN keys(n) WHERE toString(n[key]) CONTAINS $q)
			RETURN n AS node, 0.1 AS score
			LIMIT $k
			"""
			with self._driver.session(database=self.config.database) as session:
				result = session.run(fallback, q=query, k=top_k)
				records = list(result)
		documents: List[RetrievalDocument] = []
		for record in records:
			node = record["node"]
			props, text = self._format_node(node)
			if not text:
				text = props.get("summary", "") or props.get("name", "")
			documents.append(
				RetrievalDocument(
					text=text,
					score=float(record.get("score", 0.0)),
					metadata=props,
				)
			)
		return documents


__all__ = ["VietMedKGRetriever"]
