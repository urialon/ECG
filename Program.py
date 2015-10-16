import colorama
import pymongo
import DumbDiagnoser
import Common
from sklearn.neighbors import LSHForest
from Common import Constants, CreateFeatureVector

featureVectorFactor = [10.45935922250565026, 13211.087958630917, 2551.4811742410716, 921.5007903683774, 398.02065999138813, 112.54104110334723, 133.8470938825284, 650.0264814360967, 162.64549063052917, 160.09765706586313]
peopleFactor = [100, 1, 100, 10, 10, 10, 10, 100]

loadPreviousResults = False
ChartsNeighbors = 5
ChartsCandidates = 10
ChartsEstimators = 10
PeopleNeighbors = 3
PeopleCandidates = 10
PeopleEstimators = 10

client = pymongo.MongoClient(Constants.LocalHost, Constants.MongoPort)

TrainingSetSize = 70

def Main():
    trainingSet, people = LoadTrainingSet()
    # Uncomment when running from console:
    # colorama.init()
    if loadPreviousResults:
        previouslyLearnedVectors, previouslyLearnedPeople = LoadPreviouslyLearnedResults()
        trainingSet.extend(previouslyLearnedVectors)
        people.extend(previouslyLearnedPeople)
    else:
        client.drop_database(Constants.PreviousResultsDb)

    chartsForest = LSHForest(n_neighbors = ChartsNeighbors, n_estimators = ChartsEstimators, n_candidates = ChartsCandidates)
    chartsForest.fit(trainingSet)

    peopleForest = LSHForest(n_neighbors = PeopleNeighbors, n_estimators = PeopleEstimators, n_candidates = PeopleCandidates)
    peopleForest.fit(people)

    while True:
        try:
            featureVector, person = GetNewInput()
            ShowCurrentPatient(person)
            warnings = DumbDiagnoser.GetDumbDiagnosis(featureVector, person)
            diagnosis, closestChartsPeople = Diagnose(chartsForest, featureVector)
            closestPeople = GetClosestPeople(peopleForest, person)
            ShowWarnings(warnings)
            ShowResults(diagnosis, closestChartsPeople, closestPeople)
            Learn(chartsForest, featureVector, peopleForest, person, diagnosis)
        except EOFError:
            print('Exiting')
            client.close()
            break
        except NoSuchRecordException as details:
            print(details)
        finally:
            print

def ShowWarnings(warnings):
    print(colorama.Fore.YELLOW + colorama.Style.BRIGHT + 'Warnings ' + colorama.Style.RESET_ALL + '(based on raw personal information and ECG chart only):')
    for warning in warnings:
        print('\t' + warning)

def ShowCurrentPatient(person):
    print('You are:')
    print(PersonToDescription(person, False))

def ShowResults(diagnosis, closestChartsPeople, closestPeople):
    ShowClosestChartsPeople(closestChartsPeople)
    print
    print(colorama.Fore.RED + colorama.Style.BRIGHT + 'Your suggested diagnosis' + colorama.Style.RESET_ALL + ': ' + colorama.Style.BRIGHT + diagnosis + colorama.Style.RESET_ALL)
    print
    ShowClosestPeople(closestPeople)


def ShowClosestPeople(closestPeople):
    print('People who have the closest personal information are: ')
    for person in closestPeople:
        result = PersonToDescription(person, True)
        print(result)

def ShowClosestChartsPeople(closestChartsPeople):
    print('People who have the closest charts are: ')
    for person in closestChartsPeople:
        result = PersonToDescription(person, True)
        print(result)

def Learn(chartsForest, featureVector, peopleForest, person, diagnosis):
    print('Learning you and your chart...')

    person[Constants.Diagnosis] = [diagnosis]
    vectorizedChartToAdd = Common.VectorizeFeatureVector(featureVector._asdict())
    factorizedChartToAdd = Common.FactorizeVectors([vectorizedChartToAdd], featureVectorFactor)
    chartsForest.partial_fit(factorizedChartToAdd)

    vectorizedPersonToAdd = VectorizePerson(person)
    factorizedPersonToAdd = Common.FactorizeVectors([vectorizedPersonToAdd], peopleFactor)
    peopleForest.partial_fit(factorizedPersonToAdd)

    previousResultsDb = client.get_database(Constants.PreviousResultsDb)
    vectorsCollection = previousResultsDb.get_collection(Constants.FeatureVectors)
    peopleCollection = previousResultsDb.get_collection(Constants.People)
    vectorsCollection.insert_one(featureVector._asdict())
    # Allow adding a record and a person that are in the already learned collection
    # by deleting the "_id" key, which will be created automatically upon re-insertion
    idFieldName = '_id'
    if idFieldName in person:
        del person[idFieldName]
    peopleCollection.insert_one(person)


def PersonToDescription(person, shouldPrintDiagnosis):
    bmi = person[Constants.BMI]
    sysBP = person[Constants.SystolicBP]
    diBP = person[Constants.DiastolicBP]
    if person[Constants.Gender] == 1:
        gender = 'woman'
    else:
        gender = 'man'
    if (bmi < 18.5):
        weightStatus = 'underweight'
    elif (25 < bmi <= 30):
        weightStatus = 'overweight'
    elif (bmi > 30):
        weightStatus = 'obese'
    else:
        weightStatus = 'normal weight'
    result = 'A ' + str(person['Age']) + ' years old ' + weightStatus + ' ' + gender
    if (person[Constants.Smoking] == 1):
        result += ', smoking'
    else:
        result += ', not smoking'
    result += ', sport active: ' + str(person['Sport'])

    if (sysBP < 90 & diBP < 60):
        bpStatus = 'low blood pressure'
    elif ((120 < sysBP < 140) & (diBP < 90)) | ((80 < diBP < 90) & (sysBP < 140)):
        bpStatus = 'pre-high blood pressure'
    elif (sysBP > 140 | diBP > 90):
        bpStatus = 'high blood pressure'
    else:
        bpStatus = 'normal blood pressure'
    result += ', ' + bpStatus

    if (person[Constants.familyHistory] == 1):
        history = ', with '
    else:
        history = ', no '
    result += history + 'family history of cardiovascular diseases'

    if (shouldPrintDiagnosis == True):
        result += '.\n' + 'Diagnosis' + ': ' + colorama.Style.BRIGHT + (', ').join(person['Diagnosis']) + colorama.Style.RESET_ALL
    return result

def GetNewInput():
    print('Enter a new record to diagnose.')
    db, record = raw_input('Enter DB and record: ').split()
    person = GetPersonFromRecord(db, record)
    featureVector = CreateFeatureVectorFromDatabase(record, db)
    return featureVector, person

def CreateFeatureVectorFromDatabase(record, dbName):
    db = client.get_database(dbName)
    collection = db.get_collection(record)
    return CreateFeatureVector(collection, dbName, 30)

def Diagnose(chartsForest, featureVector):
    factorized = Common.FactorizeVectors([Common.VectorizeFeatureVector(featureVector._asdict())], featureVectorFactor)
    distances, indices = chartsForest.kneighbors(factorized, n_neighbors=ChartsNeighbors)
    closestChartsPeople = GetPeopleByIndices(indices[0])
    predicted = GetPrediction(distances[0], closestChartsPeople)
    return predicted, closestChartsPeople

def GetPeopleByIndices(indices):
    people = list()
    for index in indices:
        if index < TrainingSetSize:
            # The person is in the original training set
            cursor = client.get_database(Constants.TrainingSetDbName).get_collection(Constants.People).find()
        else:
            # The person is in the previously learned set
            index = index - TrainingSetSize
            cursor = client.get_database(Constants.PreviousResultsDb).get_collection(Constants.People).find()
        if (index > 0):
            cursor = cursor.skip(int(index))
        person = cursor.next()
        people.append(person)

    return people

def GetPrediction(distances, closestChartsPeople):
    # Suggest a diagnosis for a patient, based on his neighbors and their distances.
    # The most common diagnosis wins. If there are more than one with the same appearances among the neighbors,
    # the diagnosis which its sum of distances is the lowest wins
    histogram = dict()
    sumOfDistances = dict()
    for personAndDistance in zip(closestChartsPeople, distances):
        for diagnosis in personAndDistance[0][Constants.Diagnosis]:
            if histogram.has_key(diagnosis):
                histogram[diagnosis] += 1
                sumOfDistances[diagnosis] += personAndDistance[1]
            else:
                histogram[diagnosis] = 1
                sumOfDistances[diagnosis] = personAndDistance[1]

    best = histogram.keys()[0]

    for key in histogram.keys():
        if (key != best):
            if (histogram[key] > histogram[best]):
                best = key
            elif (histogram[key] == histogram[best]) & (sumOfDistances[key] < sumOfDistances[best]):
                best = key

    return best

def GetClosestPeople(peopleForest, person):
    factorized = Common.FactorizeVectors([VectorizePerson(person)], peopleFactor)
    distances, indices = peopleForest.kneighbors(factorized, n_neighbors=PeopleNeighbors)
    closestPeople = GetPeopleByIndices(indices[0])
    return closestPeople

def GetPersonFromRecord(db, record):
    peopleDb = client.get_database(Constants.People)
    peopleCollection = peopleDb.get_collection(Constants.People)
    personToRecordCollection = peopleDb.get_collection(Constants.PersonToRecordCollection)

    personToRecord = personToRecordCollection.find_one({Constants.Record: record, Constants.Database: db})
    if personToRecord is None:
        raise NoSuchRecordException('No such record as ' + record + ' in DB: ' + db)
    id = personToRecord[Constants.ID]
    person = peopleCollection.find_one({Constants.ID: id})
    return person

def LoadTrainingSet():
    trainingSetDb = client.get_database(Constants.TrainingSetDbName)
    trainingSetCollection = trainingSetDb.get_collection(Constants.TrainingSetCollectionName)
    peopleTrainingSetCollection = trainingSetDb.get_collection(Constants.People)

    vectorizedFeatureVectors = VectorizeTrainingSet(trainingSetCollection.find())
    vectorizedPeople = VectorizePeopleTrainingSet(peopleTrainingSetCollection.find())
    factorizedFeatureVectors = Common.FactorizeVectors(vectorizedFeatureVectors, featureVectorFactor)
    factorizedPeople = Common.FactorizeVectors(vectorizedPeople, peopleFactor)
    return factorizedFeatureVectors, factorizedPeople

def VectorizeTrainingSet(trainingSet):
    result = [Common.VectorizeFeatureVector(featureVector) for featureVector in trainingSet]
    return result

def VectorizePeopleTrainingSet(people):
    result = [VectorizePerson(person) for person in people]
    return result

def VectorizePerson(person):
    return [
            person[Constants.Age],
            person[Constants.Gender],
            person[Constants.Smoking],
            person[Constants.familyHistory],
            person[Constants.Sport],
            person[Constants.SystolicBP],
            person[Constants.DiastolicBP],
            person[Constants.BMI]
    ]

def LoadPreviouslyLearnedResults():
    previousResultsDb = client.get_database(Constants.PreviousResultsDb)
    vectorsCollection = previousResultsDb.get_collection(Constants.FeatureVectors)
    peopleCollection = previousResultsDb.get_collection(Constants.People)

    vectorizedFeatureVectors = VectorizeTrainingSet(vectorsCollection.find())
    vectorizedPeople = VectorizePeopleTrainingSet(peopleCollection.find())
    factorizedFeatureVectors = Common.FactorizeVectors(vectorizedFeatureVectors, featureVectorFactor)
    factorizedPeople = Common.FactorizeVectors(vectorizedPeople, peopleFactor)
    return factorizedFeatureVectors, factorizedPeople

class NoSuchRecordException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

Main()