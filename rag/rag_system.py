import os
import json
import logging
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)
load_dotenv()

class LLMAnalyzer:
    """LLM 分析器"""

    def __init__(self):
        self.api_key = os.getenv("API_TOKEN")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )

    async def analyze(self, prompt: str) -> Optional[Dict]:
        """执行 LLM 分析"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a fraud detection expert. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            content = response.choices[0].message.content
            logger.info(f"LLM raw content: {content}")
            # 剥离 markdown 代码块标记
            content = re.sub(r"^\s*```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```\s*$", "", content)
            # 转义反斜杠以修复 JSON 解析问题
            content = content.replace('\\', '\\\\')
            try:
                parsed = json.loads(content)
                logger.info(f"LLM parsed result: {parsed}")
                return parsed
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON. Raw content: {content}")
                return None
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return None

class FraudDetector:
    """诈骗检测系统"""

    def __init__(self, faiss_manager):
        self.faiss = faiss_manager
        self.scam_texts = []
        self.llm = LLMAnalyzer()
        self.KEYWORD_RULES = {
            "financial": {
                "patterns": [
                    (re.compile(r"\btransfer\b", re.IGNORECASE), 0.4),
                    (re.compile(r"\baccount\b", re.IGNORECASE), 0.3),
                    (re.compile(r"\$\d[\d,]*"), 0.5),
                    (re.compile(r"\bpayment\b", re.IGNORECASE), 0.3),
                    (re.compile(r"\bmoney\b", re.IGNORECASE), 0.4),
                    (re.compile(r"\bbank\b", re.IGNORECASE), 0.3)
                ],
                "max_score": 0.7
            },
            "urgency": {
                "patterns": [
                    (re.compile(r"\burgen(t|cy)\b", re.IGNORECASE), 0.5),
                    (re.compile(r"\bimmediately\b", re.IGNORECASE), 0.4),
                    (re.compile(r"\blimited time\b", re.IGNORECASE), 0.4),
                    (re.compile(r"\bexpire[sd]?\b", re.IGNORECASE), 0.4),
                    (re.compile(r"\bquick(ly)?\b", re.IGNORECASE), 0.3)
                ],
                "max_score": 0.6
            },
            "suspicious": {
                "patterns": [
                    (re.compile(r"\bverify\b", re.IGNORECASE), 0.5),
                    (re.compile(r"\bclick\b", re.IGNORECASE), 0.4),
                    (re.compile(r"\bsecure\b", re.IGNORECASE), 0.3),
                    (re.compile(r"\bconfirm\b", re.IGNORECASE), 0.4),
                    (re.compile(r"\bvalidate\b", re.IGNORECASE), 0.4)
                ],
                "max_score": 0.6
            }
        }

    def load_scam_dataset(self, texts: List[str]):
        self.scam_texts = texts
        logger.info(f"Loaded {len(texts)} scam text samples")

    def _generate_report(self, text: str, patterns: Dict, similar: List, llm: Dict) -> str:
        """Generate a concise analysis report, maintaining user-friendliness"""
        report_parts = []
        
        # Add risk assessment
        risk_level = "High Risk" if patterns["total_score"] > 0.7 else "Medium Risk" if patterns["total_score"] > 0.4 else "Low Risk"
        report_parts.append(f"**Risk Level**: {risk_level}")
        
        # Add suspicious patterns
        if patterns["categories"]:
            suspicious_patterns = []
            for category, details in patterns["categories"].items():
                if details["score"] > 0:
                    if category == "financial":
                        suspicious_patterns.append("Mentions of banks, amounts, accounts, etc.")
                    elif category == "urgency":
                        suspicious_patterns.append("Urges immediate action or response within a limited time")
                    elif category == "suspicious":
                        suspicious_patterns.append("Contains suspicious words like 'click', 'secure', etc.")
            if suspicious_patterns:
                report_parts.append("**Suspicious Patterns**  \n- " + "  \n- ".join(suspicious_patterns))
        
        # Add similar cases (limit to top 2)
        if similar:
            similar_cases = []
            for case in similar[:2]:  # Show only the top two similar cases
                similarity = round(case["similarity"] * 100, 1)
                if similarity > 30:  # Only show cases with similarity > 30%
                    similar_cases.append(f"Similarity {similarity}%: {case['text'][:50]}...")
            if similar_cases:
                report_parts.append("**Similar Cases**  \n- " + "  \n- ".join(similar_cases))
        
        # Add LLM analysis results (if available)
        if llm and not llm.get("error"):
            if llm.get("indicators"):
                indicators = [indicator for indicator in llm["indicators"] if indicator.strip()]
                if indicators:
                    report_parts.append("**Risk Indicators**  \n- " + "  \n- ".join(indicators))
            if llm.get("recommendations"):
                recommendations = [rec for rec in llm["recommendations"] if rec.strip()]
                if recommendations:
                    report_parts.append("**Recommended Actions**  \n- " + "  \n- ".join(recommendations))
        
        return "\n\n".join(report_parts)

    async def analyze_text(self, text: str) -> Dict:
        if not text or len(text.strip()) < 3:
            return {
                "risk_score": 0,
                "risk_level": "Invalid Input",
                "pattern_analysis": {"categories": {}, "total_score": 0},
                "similar_cases": [],
                "llm_analysis": {"error": "Text too short"},
                "report": "Text too short"
            }
        pattern_results = self._detect_risk_patterns(text)
        similar_cases = self._retrieve_similar_cases(text)
        llm_report = await self._generate_llm_report(text, pattern_results, similar_cases)
        risk_score = self._calculate_risk_score(pattern_results, similar_cases, llm_report)
        report_text = self._generate_report(text, pattern_results, similar_cases, llm_report)
        return {
            "risk_score": risk_score,
            "risk_level": self._determine_risk_level(risk_score),
            "pattern_analysis": pattern_results,
            "similar_cases": similar_cases,
            "llm_analysis": llm_report,
            "report": report_text
        }

    def _detect_risk_patterns(self, text: str) -> Dict:
        results = {"categories": {}, "total_score": 0.0}
        text_lower = text.lower()
        for category, config in self.KEYWORD_RULES.items():
            category_score = 0.0
            detected_patterns = []
            for pattern, weight in config["patterns"]:
                if pattern.search(text_lower):
                    category_score = max(category_score, weight)
                    detected_patterns.append(pattern.pattern)
            final_score = min(category_score, config["max_score"])
            results["categories"][category] = {
                "score": final_score,
                "matched_patterns": list(set(detected_patterns))
            }
        total = sum(v["score"] for v in results["categories"].values())
        results["total_score"] = round(total / len(self.KEYWORD_RULES), 4)
        logger.info(f"Pattern score for text '{text}': {results['total_score']}, details: {results['categories']}")
        return results

    def _retrieve_similar_cases(self, text: str, top_k: int = 3) -> List[Dict]:
        try:
            search_results = self.faiss.search_similar(text, top_k)
            valid_results = []
            for result in search_results:
                idx = result.get("index", -1)
                if 0 <= idx < len(self.scam_texts):
                    valid_results.append({
                        "text": self.scam_texts[idx],
                        "similarity": round(result.get("score", 0), 4)
                    })
                    logger.debug(f"Found similar: {self.scam_texts[idx]} with score: {result.get('score', 0)}")
            return valid_results
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []

    async def _generate_llm_report(self, text: str, patterns: Dict, similar: List) -> Dict:
        try:
            pattern_summary = []
            for category, details in patterns["categories"].items():
                if details["score"] > 0:
                    pattern_summary.append(f"{category}: {', '.join(details['matched_patterns'])}")
            
            prompt = f"""Analyze the potential fraud risk of the following text:

Text: {text}

Detected patterns:
{chr(10).join(pattern_summary) if pattern_summary else 'No suspicious patterns detected'}

Please provide the analysis in the following JSON format:
{{
    "risk_score": (risk score between 0 and 1),
    "indicators": [list of main risk indicators],
    "recommendations": [recommended actions],
    "confidence": "Low/Medium/High"
}}"""

            analysis = await self.llm.analyze(prompt)
            if not analysis:
                return {
                    "error": "LLM analysis unavailable",
                    "risk_score": 0,
                    "indicators": [],
                    "recommendations": ["Please rely on pattern matching results"],
                    "confidence": "Low"
                }
            
            # 清理和验证 LLM 返回的数据
            return {
                "risk_score": float(analysis.get("risk_score", 0)),
                "indicators": [str(i).strip() for i in analysis.get("indicators", []) if i],
                "recommendations": [str(r).strip() for r in analysis.get("recommendations", []) if r],
                "confidence": str(analysis.get("confidence", "Low"))
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return {
                "error": "LLM analysis failed",
                "risk_score": 0,
                "indicators": [],
                "recommendations": ["Please rely on pattern matching results"],
                "confidence": "Low"
            }

    def _calculate_risk_score(self, patterns: Dict, similar: List, llm: Dict) -> float:
        weights = {"pattern": 0.4, "similarity": 0.3, "llm": 0.3}
        similarity_score = max([case["similarity"] for case in similar], default=0)
        llm_score = llm.get("risk_score", 0) if "error" not in llm else 0
        return round(
            (patterns["total_score"] * weights["pattern"]) +
            (similarity_score * weights["similarity"]) +
            (llm_score * weights["llm"]), 4
        )

    def _determine_risk_level(self, score: float) -> str:
        if score >= 0.7:
            return "Critical Risk"
        elif score >= 0.5:
            return "High Risk"
        elif score >= 0.3:
            return "Medium Risk"
        return "Low Risk"