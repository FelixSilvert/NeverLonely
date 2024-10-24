import random as r

def surpriseCharacter():
    sex = ["male", "women"]
    hair = ["blond", "brown", "black", "redhead"]
    eyes = ["blue", "brown", "yellow", "green", "grey", "black"]
    age = ["young", "adult", "old"]
    skinColor = ["white", "black", "asian", "arabian", "indian"]
    hairStyle = ["straight", "curly", "wavy"]

    val1 = r.randint(0,1)
    val2 = r.randint(0,3)
    val3 = r.randint(0,5)
    val4 = r.randint(0,2)
    val5 = r.randint(0,4)
    val6 = r.randint(0,2)

    return skinColor[val5] + " " + age[val4] + " " + sex[val1] + ", with " + hair[val2] + " " + hairStyle[val6] + " " + "hair," + " " + eyes[val3] + " " + "eyes"
