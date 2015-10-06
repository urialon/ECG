from Common import Constants

def GetDumbDiagnosis(featureVector, person):
    warnings = list()
    warnings += DumbChartDiagnoser(featureVector)
    warnings += DumbPersonDiagnoser(person)
    return warnings


def DumbChartDiagnoser(featureVector):
    warnings = list()
    heartbeat = featureVector.AverageHeartbeat
    if heartbeat > 100:
        warnings.append('Tachycardia - Heart rate is higher than normal rate')
    if heartbeat < 60:
        warnings.append('Bradycardia - Heart rate is lower than normal rate')

    if featureVector.IrregularBeatsPercent > 0.05:
        warnings.append('Arrhythmia - Irregular heart rate')

    if featureVector.QtoR > float(1) / 3:
        warnings.append('Deep Q wave, might suggest Myocardial infarction')

    if featureVector.StoR > float(1) / 3:
        warnings.append('Deep S wave, might suggest Ventricular Hypertrophy')
    return warnings


def DumbPersonDiagnoser(person):
    warnings = list()

    bmi = person[Constants.BMI]
    sysBP = person[Constants.SystolicBP]
    diBP = person[Constants.DiastolicBP]
    if (bmi < 18.5):
        warnings.append('Low BMI - underweight')
    if (25 < bmi <= 30):
        warnings.append('High BMI - overweight')
    if (bmi > 30):
        warnings.append('Very high BMI - Obese')
    if (sysBP < 90 & diBP < 60):
        warnings.append('Too low blood pressure')
    if ((120 < sysBP < 140) & (diBP < 90)) | ((80 < diBP < 90) & (sysBP < 140)):
        warnings.append('Pre-high blood pressure')
    if (sysBP > 140 | diBP > 90):
        warnings.append('High blood pressure')

    if (person[Constants.familyHistory] == 1) & \
            ((person[Constants.Smoking] == 1) | (person[Constants.Age] > 40) | (bmi > 25) | (sysBP > 120) | (diBP > 80)):
        warnings.append('High risk for cardiovascular disease')
    return warnings
