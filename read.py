#%%

import json
import collections
import pandas as pd

class Read_Process(object):
    def __init__(self,filename):
        self.dic = collections.defaultdict(list)
        self.Gaode_Type = pd.read_excel("高德poi地址类别全集.xlsx")
        self.filename = filename


    def Read_Data(self):
        """
        读取去重
        """
        with open(self.filename, encoding="utf-8") as f:
            # id = set()
            for line in f.readlines():
                j = json.loads(line)
                for i in j:
                    # id.add(i[0])
                    self.dic["id"].append(i[0])
                    self.dic["pname"].append(i[1])
                    self.dic["cityname"].append(i[2])
                    self.dic["typecode"].append(i[3])
                    self.dic["name"].append(i[4])
                    self.dic["`address`"].append(i[5])
                    self.dic["region"].append(i[6])
                    self.dic["tel"].append((i[7]))
                    self.dic["gcj02_lng"].append(i[8])
                    self.dic["gcj02_lat"].append(i[9])
        f.close()
        data = pd.DataFrame(self.dic)
        for col in data.columns:
            data[col] = data[col].astype("str")
        data.drop_duplicates(inplace=True)
        return data
        # self.data =060700,170000,170200,


    #
    def _GetPhone(self,data):
        """
        电话号处理
        :return:
        """
        # data = self.Read_Data()
        data['tel_process'] = data['tel'].map(lambda x: x.split(';'))
        data = data.explode('tel_process')

        # 手机号 + 电话号
        def get_ceil_phone(x):
            if len(x) == 11 and x[0] == "1":
                return x

        def get_phone(x):
            if len(x) > 3 and len(x) != 11:
                return x

        data["mobel_phone"] = data.tel_process.apply(lambda x: get_ceil_phone(x))
        data["phone"] = data.tel_process.apply(lambda x: get_phone(x))
        return data


    def Process(self):
        """

        :return:
        """
        data  = self.Read_Data()
        def type_process(x):
            """
            类别处理
            """
            x = x.split("|")
            for i in x:
                if i not in ["060700","170000","170200"]:
                    return int(i)
            return (x[-1])
        #
        data["typecode"] = data.typecode.apply(lambda x: type_process(x))
        data = data.merge(self.Gaode_Type[["NEW_TYPE", "小类"]], how="left", left_on="typecode",
                                            right_on="NEW_TYPE")
        data.rename({"小类":"label"},axis = 1,inplace=True)
        return data

if __name__=='__main__':
    R = Read_Process("spider_v2.txt")
    data = R.Process()
    # data.to_csv("processed.csv",index=False)