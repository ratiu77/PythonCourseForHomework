#Scrieti un test cu `pytest` pentru acest agent, puteti doar sa rulati simplu unealta, si sa verificati ca agentul a returnat text care contine numele unui fisier la care ne asteptam sa contina acel string pytest. puteti verifica daca textul returnat de agent contine un substring, folosind : "if 'substring' in given_string", unde given_string poate fi intregul text returnat de agent.

from tema2 import find_keyword


def test_find_keyword():
    given_string = find_keyword()

    if "tema2.py" in given_string:
        print("Am gasit tema2.py")
    else:
        print("Nu am gasit tema2.py")