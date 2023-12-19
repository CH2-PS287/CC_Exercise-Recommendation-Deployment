def calculate_bmr(weight_kg, height_cm, age, gender):
    if gender == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif gender == 'female':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        raise ValueError("Invalid gender")
    return bmr

def calculate_tdee(bmr, activity_factor):
    tdee = bmr * activity_factor
    return tdee