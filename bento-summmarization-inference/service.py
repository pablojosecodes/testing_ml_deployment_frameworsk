from __future__ import annotations
import bentoml
from transformers import pipeline


EXAMPLE_INPUT = """Franklin organized a group of friends to provide a structured form of mutual improvement. 
The group, initially composed of twelve members, called itself the Junto (from the Spanish word junta, or assembly). 
The members of the Junto were drawn from diverse occupations and backgrounds, but they all shared a spirit of 
inquiry and a desire to improve themselves, their community, and to help others.
Among the original members were printers, surveyors, a cabinetmaker, a clerk, and a bartender. 
Although most of the members were older than Franklin, he was clearly their leader.
"""


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class Summarization:
    def __init__(self) -> None:
        # Load model into pipeline
        self.pipeline = pipeline('summarization')
    
    @bentoml.api
    def summarize(self, text: str = EXAMPLE_INPUT) -> str:
        result = self.pipeline(text)
        return result[0]['summary_text']