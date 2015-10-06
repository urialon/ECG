import Common
from pymongo import MongoClient
import pymongo
from Common import SamplingRates, Constants, Person
import random

def PopulateRandomPeople():
    currentId = 1
    client = MongoClient(Constants.LocalHost, Constants.MongoPort)
    db = client.get_database(Constants.FeatureVectors)
    client.drop_database(Constants.People)
    peopleDb = client.get_database(Constants.People)
    peopleCollection = peopleDb.get_collection(Constants.People)
    personToRecord = peopleDb.get_collection(Constants.PersonToRecordCollection)

    for singleCollection in SamplingRates.keys():
        collection = db.get_collection(singleCollection)
        for vector in collection.find():
            if (singleCollection == 'mitdb'):
                diagnosis = ['Arrhythmia']
            elif (singleCollection == 'svdb'):
                diagnosis = ['Supraventricular Arrhythmia']
            elif (singleCollection == 'afdb'):
                diagnosis = ['Atrial Fibrillation']
            elif (singleCollection == 'nsrdb'):
                diagnosis = ['Normal Sinus Rhythm']
            elif (singleCollection == 'cudb'):
                diagnosis = ['Ventricular Tachyarrhythmia']

            newPerson = CreateNewRandomPerson(currentId, diagnosis)
            currentId += 1
            print(newPerson)
            print('Assigned to record ' + vector[Constants.RecordNumber].__str__() + ' in DB: ' + vector[Constants.Database].__str__())
            peopleCollection.insert_one(newPerson._asdict())
            personToRecord.insert_one({Constants.ID: newPerson.ID, 'Database': vector[Constants.Database], Constants.Record: vector[Constants.RecordNumber]})
    peopleCollection.create_index(Constants.ID)
    personToRecord.create_index(Constants.ID)
    personToRecord.create_index([(Constants.Database, pymongo.ASCENDING), (Constants.Record, pymongo.ASCENDING)])

def CreateNewRandomPerson(id, diagnosis):
    result = Person(
        ID = id,
        Gender = random.randint(0, 1) ,
        Smoking = random.randint(0, 1),
        familyHistory = random.randint(0, 1),
        Sport = random.randint(0, 10),
        Age = random.randint(18, 65),
        SystolicBP = random.randint(70, 190),
        DiastolicBP = random.randint(40, 100),
        BMI = random.uniform(17, 31),
        Diagnosis = diagnosis
    )
    return result

def PopulatePeople():
    PopulateRandomPeople()

def PopulateFeatureVectors():
    takeFirstMinutes = 30
    client = MongoClient(Constants.LocalHost, Constants.MongoPort)
    client.drop_database(Constants.FeatureVectors)
    vectorsDb = client.get_database(Constants.FeatureVectors)

    for dbName in SamplingRates.keys():
        db = client.get_database(dbName)
        vectorsCollection = vectorsDb.get_collection(dbName)
        for collectionName in db.collection_names():
            if (collectionName.startswith('system')):
                continue
            collection = db.get_collection(collectionName)
            print('DB: ' + dbName + ', collection: ' + collectionName)
            result = Common.CreateFeatureVector(collection, dbName, takeFirstMinutes)
            print(result)
            vectorsCollection.insert_one(result._asdict())

def CreateTrainingSetDatabase():
    Common.PutTrainingSet(Common.GetTrainingSet())
    Common.PutPeopleTrainingSet(Common.GetPeopleTrainingSet())

def SetUp():
    PopulateFeatureVectors()
    PopulatePeople()
    CreateTrainingSetDatabase()

SetUp()