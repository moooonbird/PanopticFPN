
import numpy
import pickle

extLabelData = "pkl"

ID_DONT_CARE = 'DontCare'
ID_DOUBLE_ROOT = 'DoubleRoot'

ID_COLUMN = "Column"
ID_ROW = "Row"
ID_MAP_PIXEL2REGIONID = "MapPixel2RegionId"
ID_LIST_REGION = "ListRegion"

'''
    ROIData 紀錄某個 id 的 ROI 的屬性
'''
class ROIData:
    def __init__(self, idROI, listRegionId):
        self.idROI = idROI
        self.listRegionId = listRegionId
        self.dictAttribute = { ID_DONT_CARE:False , ID_DOUBLE_ROOT:False}

'''
    LabelData 紀錄圖片標記結果
'''
class LabelData:
    def __init__(self, cntImageColumn, cntImageRow):
        self.cntImageColumn = cntImageColumn
        self.cntImageRow = cntImageRow
        self.mapPixel2RegionId = numpy.zeros((cntImageRow, cntImageColumn), numpy.int32)
        self.listROI = []

    def ApplyReIdMap(self, mapReId):
        for i in range(len(self.listROI)):
            roiCurrent = self.listROI[i]
            listRegionId = []
            for j in range(len(roiCurrent.listRegionId)):
                listRegionId.extend(mapReId[roiCurrent.listRegionId[j]])
            
            roiCurrent.listRegionId = listRegionId
    
    def LoadFromFile(self, path):
        fileData = open(path, "rb")
        dictData = pickle.load(fileData)
        fileData.close()

        # extract from dict
        if(ID_COLUMN in dictData and ID_ROW in dictData):
            cntImageColumn = dictData[ID_COLUMN]
            cntImageRow = dictData[ID_ROW]
            if(self.cntImageColumn != cntImageColumn or self.cntImageRow != cntImageRow):
                print("image width and height not match, label data broken!")
                return
        else:
            print("image width not found, label data broken!")
            return
        
        if(ID_MAP_PIXEL2REGIONID in dictData):
            self.mapPixel2RegionId = dictData[ID_MAP_PIXEL2REGIONID]
        else:
            print("region id record not founrd, label data broken!")
        
        if(ID_LIST_REGION in dictData):
            self.listROI = dictData[ID_LIST_REGION]
        else:
            print("region attribute record not founrd, label data broken!")
    
    def SaveToFile(self, path):
        fileData = open(path, "wb")

        # zip into dict
        dictData = {ID_COLUMN:self.cntImageColumn, ID_ROW:self.cntImageRow, ID_MAP_PIXEL2REGIONID:self.mapPixel2RegionId, ID_LIST_REGION:self.listROI}

        pickle.dump(dictData, fileData)
        fileData.close()



