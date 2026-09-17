import json
from pydantic import BaseModel, EmailStr, Field, ValidationError


with open("student.json", "r",encoding="utf-8") as f:
    data = json.load(f)



class StudentValidator(BaseModel):
    nume: str
    varsta: int = Field(gt=0)
    email: EmailStr

try:
    validated_user = StudentValidator.model_validate(data, strict=True)

    print(validated_user)
except ValidationError as e:
    print(e)
    print(e.errors("Eroare in student.json"))
finally:
    print("am terminat valiadarea")