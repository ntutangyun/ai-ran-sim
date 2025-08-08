from dataclasses import dataclass
import json

from pydantic import BaseModel, Field


class Message:
    def __init__(self, role: str, content: str):
        self.role: str = role
        self.content: str = content

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content
        }

class Data:
    def __init__(self, current_step: str, messages: list[Message]):
        self.current_step = current_step
        self.messages = messages

    def to_dict(self):
        return {
            "current_step": self.current_step,
            "messages": [message.to_dict() for message in self.messages]
        }

class Request:
    def __init__(self, layer: str, command: str, data: Data):
        self.layer = layer
        self.command = command
        self.data = data

    def to_dict(self):
        return {
            "layer": self.layer,
            "command": self.command,
            "data": self.data.to_dict()
        }

class TestOutput(BaseModel):
    answer: str | None = Field(description="The answer or clarification on requirement to the chatbot's conversation")
    isDone: bool = Field(description="If the chatbot has suggested an appropriate model, this field is marked as true, if not leave blank")
    suggested_model: str | list[str] | None = Field("The model/models suggested by the chatbot for the requirement")


class Output(BaseModel):
    scenario: str
    detailed_scenario: str
    ground_truth: str
    suggetion: str
    score: int

    @classmethod
    def get_field_names(cls):
        return list(cls.model_fields.keys())