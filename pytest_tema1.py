import pytest
from pydantic import ValidationError, EmailStr

from Tema1 import StudentValidator


def test_student():
    student = {

    "nume": "Ana Popescu",
    "varsta": 20,
    "email": "ana.popescu"

    }
    with pytest.raises(ValidationError):
        student1 = StudentValidator.model_validate(student, strict=True)

        assert isinstance(student1, StudentValidator)
        assert len(student1.nume) <= 50 and len(student1.nume) >= 2
        assert isinstance(student1.nume, str)
        assert isinstance(student1.varsta, int)
        assert student1.varsta <= 100 and student1.varsta >= 17


def test_student1():

    student = {

    "nume": "Ana Popescu",
    "varsta": 20,
    "email": "ana.popescu@example.com"

    }
    try:

            student1 = StudentValidator.model_validate(student, strict=True)

            assert isinstance(student1, StudentValidator)
            assert len(student1.nume) <= 50 and len(student1.nume) >= 2
            assert isinstance(student1.nume, str)
            assert isinstance(student1.varsta, int)
            assert student1.varsta <= 100 and student1.varsta >= 17
    except ValidationError as e:
        print(e)

