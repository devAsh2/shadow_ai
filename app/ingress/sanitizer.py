##Sanitizer service 
#-----
# Masks sensitive information, generates a unique session key, and stores the mapping in Redis with a 60-second TTL
import secrets
import json
import redis.asyncio as redis
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from app.ingress.custom_nlp import NoOpNlpEngiine

class SanitizerService:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.analyzer = AnalyzerEngine(nlp_engine=NoOpNlpEngine(), supported_languages=["en"])
        self.anonymizer = AnonymizerEngine()
        self._register_custom_recognizers()

    def _register_custom_recognizers(self):
        #Email Recognizers
        email_pattern = Pattern("email_regex", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", 0.9)
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(supported_entity="EMAIL_ADDRESS", patterns=[email_pattern])
        )
        #Phone number recognizers
        #   (International + Indian formats)
        phone_pattern = Pattern("phone_regex", r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", 0.9)
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[phone_pattern])
        )

    async def sanitize_and_store(self, prompt:str) -> dict:
        session_key = secrets.token_hex(16)
        results = self.analyzer.analyze(text=prompt, entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], language="en")

        #construct the reverse map 
        pii_map = {}
        for r in results:
            original_value = prompt[r.start:r.end]
            pii_map[f"<{r.entity_type}>"] = original_value

        anonymized_result = self.anonymizer.anonymize(text=prompt, analyzer_results=results)
        masked_prompt = anonymized_result.text

        # store the reverse map in redis for easy lookups with 60 secs ttl
        if pii_map:
            redis_key = f"pii_map:{session_key}"
            await self.redis.set(redis_key, 60, json.dumps(pii_map))

        #return the dictionary output with session key, original prompt, masked prompt, and a flag indicating if PII was found
        return {
            "session_key": session_key,
            "original_prompt": prompt,
            "masked_prompt": masked_prompt,
            "has_pii": len(pii_map) > 0,
            "TTL": 60
        }