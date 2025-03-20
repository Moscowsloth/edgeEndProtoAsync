class UpdateInfo(object):
    def __init__(self, proto, index):
        self.proto = proto
        self.index = index

    def getProto(self):
        return self.proto

    def getIndex(self):
        return self.index
