##Sanitizer service 
#-----
# Masks sensitive information, generates a unique session key, and stores the mapping in Redis with a 60-second TTL
import secrets
import json
import redis.asyncio as redis
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from app.ingress.custom_nlp import DummyNlpEngine

class SanitizerService:

    # Define entities once at module or class level
    SUPPORTED_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER"]

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.analyzer = AnalyzerEngine(nlp_engine=DummyNlpEngine(), supported_languages=["en"])
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
        phone_pattern = Pattern("phone_regex", r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", 0.4)
        phone_context = [
            "phone", "call", "mobile", "cell", "contact", 
            "tel", "dial", "whatsapp", "reach"
        ]
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[phone_pattern], context=phone_context)
        )

    async def sanitize_and_store(self, prompt:str) -> dict:
        session_key = secrets.token_hex(16)
        results = self.analyzer.analyze(text=prompt, entities=self.SUPPORTED_ENTITIES, language="en", score_threshold=0.6)

        #construct the reverse map 
        # pii_map = {}
        # for r in results:
        #     original_value = prompt[r.start:r.end]
        #     pii_map[f"<{r.entity_type}>"] = original_value  --> loophole (unhandled duplicate types and overwrites)
        #Fix: reverse map tracking and ordering of entities 

        #sorted the unordered results based on start position
        ltr_results = sorted(results,key=lambda x:x.start)

        reverse_map = {}
        entity_counts = {}
        #looping sorted list, tracking their exact count and constructing maps
        for i,r in enumerate(ltr_results):
            #store entity types in counts based on counts
            count = entity_counts.get(r.entity_type,0) + 1
            entity_counts[r.entity_type] = count

            #build the name tag
            placeholder = f"<{r.entity_type}_{count}>"

            original_value = prompt[r.start:r.end]
            reverse_map[placeholder] = original_value

        # 3. Anonymizer Countdown Operator
        # We copy the totals dictionary so each entity starts at its maximum count
        current_entity_count = entity_counts.copy()

        #factory function to return a custom type placeholder 
        def custom_mask(entity_type:str):
            """Instead of manually defining 10 def for each entities, we define a factory closure function which is callable later."""
            def _mask(text_to_mask:str)->str:
                #increment counter logic
                count = current_entity_count[entity_type]
                current_entity_count[entity_type]-=1
                return f"<{entity_type}_{count}>"
            return _mask
        

        # Generate all operator configs dynamically in 1 clean line
        operators = {
            entity: OperatorConfig("custom", {"lambda": custom_mask(entity)})
            for entity in self.SUPPORTED_ENTITIES
        }
        
        anonymized_result = self.anonymizer.anonymize(text=prompt, analyzer_results=ltr_results, operators=operators)
        masked_prompt = anonymized_result.text

        # store the reverse map in redis for easy lookups with 60 secs ttl
        if reverse_map:
            redis_key = f"rev_map_session:{session_key}"
            await self.redis.set(redis_key, json.dumps(reverse_map), ex=60)

        #return the dictionary output with session key, original prompt, masked prompt, and a flag indicating if PII was found
        return {
            "session_key": session_key,
            "masked_prompt": masked_prompt,
            "has_pii": len(reverse_map) > 0,
            "TTL": 60
        }