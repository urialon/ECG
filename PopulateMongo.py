import Common
from Common import Constants
from pymongo import MongoClient, ASCENDING

takeFirstMinutes = 40
lookaheadThreshold = 4

def PopulateMongo(databasesTextPath):
    client = MongoClient(Constants.LocalHost, Constants.MongoPort)
    for databaseName in Common.SamplingRates.keys():
        maxSampleTime = Common.SamplingRates[databaseName] * 60 * takeFirstMinutes
        currentDb = client.get_database(databaseName)
        for record in getAllRecordNames(databaseName, databasesTextPath):
            if record.isspace():
                continue
            ecgRecord = createCombinedRecord(databasesTextPath, databaseName, record, maxSampleTime)
            AnnotateQS(ecgRecord)
            currentRecordCollection = currentDb.get_collection(record)
            currentRecordCollection.insert_many(ecgRecord)
            currentRecordCollection.create_index(Constants.Time)
            currentRecordCollection.create_index(Constants.Label)
            print('Done inserting record ' + record.__str__() + ' in DB: ' + databaseName)
        print('Done populating DB: ' + databaseName)





def getAllRecordNames(databaseName, databasesTextPath):
    with open(databasesTextPath + databaseName + '\RECORDS') as recordsFile:
        records = [singleRecord.strip('\n') for singleRecord in recordsFile.readlines()]
    return records

def createCombinedRecord(databaseTextPath, databaseName, record, maxSampleTime):
    recordList = list()
    signalFilePath = databaseTextPath + databaseName + '\\' + record + '.' + 'signal'
    annotationFilePath = databaseTextPath + databaseName + '\\' + record + '.' + 'annotation'
    with open(signalFilePath) as signalFile, open(annotationFilePath) as annotationFile:
        matchedPreviousLine = True
        annotationFile.readline() # skip first annotation

        for signalLine in signalFile:
            if (matchedPreviousLine):
                annotationLine = annotationFile.readline()
                if annotationLine.isspace() | (not annotationLine):
                    annotationItems = ['', '', '']
                else:
                    annotationItems = annotationLine.split()
                matchedPreviousLine = False
            signalItems = signalLine.split()
            currentAnnotation = Constants.LabelNone
            if signalItems[0] == annotationItems[1]:
                currentAnnotation = Constants.LabelR
                matchedPreviousLine = True
            if int(signalItems[0]) > maxSampleTime:
                break
            recordList.append({Constants.Time:int(signalItems[0]), Constants.Value:int(signalItems[1]), Constants.Label:currentAnnotation})
    return recordList

def AnnotateQS(ecgRecord):
    i = 0
    while i < len(ecgRecord):
        if ecgRecord[i][Constants.Label] == Constants.LabelR:
            newR = TuneR(ecgRecord, i)
            i = newR
            Qindex = FindNextCriticalPoint(ecgRecord, newR, Constants.Left, Constants.MinPoint)
            Sindex = FindNextCriticalPoint(ecgRecord, newR, Constants.Right, Constants.MinPoint)
            if ((Qindex < len(ecgRecord)) & (ecgRecord[Qindex][Constants.Label] == Constants.LabelNone)):
                ecgRecord[Qindex][Constants.Label] = Constants.LabelQ
            if ((Sindex < len(ecgRecord)) & (ecgRecord[Sindex][Constants.Label] == Constants.LabelNone)):
                ecgRecord[Sindex][Constants.Label] = Constants.LabelS
        i += 1

def TuneR(ecgRecord, index):
    tunedRightR = FindNextCriticalPoint(ecgRecord, index, Constants.Right, Constants.MaxPoint)
    tunedLeftR = FindNextCriticalPoint(ecgRecord, index, Constants.Left, Constants.MaxPoint)
    if ((tunedLeftR != index) & (tunedRightR != index)):
        if ecgRecord[tunedLeftR][Constants.Value] > ecgRecord[tunedRightR][Constants.Value]:
            tunedRightR = index
        else:
            tunedLeftR = index
    if (tunedRightR != index):
        ecgRecord[tunedRightR][Constants.Label] = Constants.LabelR
        ecgRecord[index][Constants.Label] = Constants.LabelNone
        return tunedRightR
    if (tunedLeftR != index):
        ecgRecord[tunedLeftR][Constants.Label] = Constants.LabelR
        ecgRecord[index][Constants.Label] = Constants.LabelNone
        return tunedLeftR
    return index # if not tuned

def FindNextCriticalPoint(ecgRecord, index, direction, criticalPointType):
    found = True
    currentNeighborIndex = index
    while (found):
        found = False
        for i in range(1, lookaheadThreshold+1):
            if direction == Constants.Right:
                currentNeighborIndex = index + i
            else:
                currentNeighborIndex = index - i
                if currentNeighborIndex < 0:
                    return 0
            if currentNeighborIndex < len(ecgRecord):
                if ((criticalPointType == Constants.MaxPoint) & (ecgRecord[currentNeighborIndex][Constants.Value] > ecgRecord[index][Constants.Value])) | ((criticalPointType == Constants.MinPoint) & (ecgRecord[currentNeighborIndex][Constants.Value] < ecgRecord[index][Constants.Value])):
                    index = currentNeighborIndex
                    found = True
                    break
    return index


