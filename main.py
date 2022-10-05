# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import requests
import queue
import json

from collections import defaultdict
import random
import re
import sys




class GaoDePageInfo(object):


    def __init__(self,content):
        self._fileds = json.loads(content)


    def __getattr__(self,key):
        if key in self._fileds:
            return self._fileds[key]
        return super(GaoDePageInfo,self).__getattr__(self,key)


    def to_csv_line(self):
        pois = self.pois
        rows = []
        for i in range(len(pois)):
            ID = pois[i]['id']
            pname = pois[i]['pname']
            cityname = pois[i]['cityname']
            adname = pois[i]['adname']
            name = pois[i]['name']
            location = pois[i]['location']
            typecode = pois[i]['typecode']
            tel = pois[i]['tel']

            if 'address' not in pois[i].keys():
                address = str(-1)
            else:
                address = pois[i]['address']
            result = location  # gcj02towgs84(location)
            lng = float(result.split(",")[0])
            lat = float(result.split(",")[1])
            row = [ID, pname, cityname, typecode, name, address, adname, tel, lng, lat]
            rows.append(row)
        return json.dumps(rows)


class GaodeKey(object):

    def __init__(self,keys,limit):
        self._key_dict = defaultdict(int)
        self.keys = keys
        self.limit = limit
        self._count = 0

    def get_key(self):
        self._count += 1
        if self._count % 500 == 0:
            self.dumps()

        for i in range(100):
            key = self.keys[random.randint(0,len(self.keys)-1)]
            if self._key_dict.get(key,0) < self.limit:
                self._key_dict[key]+=1
                print(self.get_status())
                return key

        return None

    def loads(self):
        pass

    def get_status(self):
        s = []
        for key in self.keys:
            s.append("{}:{}".format(key,self.limit - self._key_dict[key]))
        return "\t".join(s)

    def dumps(self):
        pass



class GaodeLink(object):
    def __init__(self,url,page_num,*args):
        self.url = url
        self.currentMinLat = None
        self.currentMaxLat = None
        self.currentMaxLon = None
        self.currentMinLon = None
        self.level = None
        self.page_num = page_num


    def __setattr__(self, key, value):
        if key.startswith("current") and value is not None:
            super(GaodeLink, self).__setattr__(key,round(value,6))
        else:
            super(GaodeLink, self).__setattr__(key,value)


    def contains(self,crawled_set):
        ## url不同key，
        return True if self.url is None else self.url in crawled_set

class GaodeSpider(object):

    def __init__(self,city_name, keywords,tokens ,spder_info_file = "./spider.txt"):
        self._keywords = keywords
        self._crawled = set()
        self._key = tokens
        self._crawl_queue = queue.Queue()
        self._spider_info_file = spder_info_file
        self._header = {
            'User-Agent': "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50"}
        _links = self._init_link(city_name)
        for link in _links:
            self._crawl_queue.put(link)



    def _init_link(self,city_name):
        parameters = 'key={}&keywords={}&subdistrict={}&output=JSON&extensions=all'.format(self._key.get_key(), city_name, 0)
        url = 'https://restapi.amap.com/v3/config/district?'
        # 设置header
        res = requests.get(url, params=parameters)
        jsonData = res.json()
        if jsonData['status'] == '1':
            district = jsonData['districts'][0]['polyline']
            district_list = re.split(';|\|', district)
            xlist, ylist = [], []
            for d in district_list:
                xlist.append(float(d.split(',')[0]))
                ylist.append(float(d.split(',')[1]))
            cityMaxLat = max(xlist)
            cityMinLat = min(xlist)
            cityMaxLon = max(ylist)
            cityMinLon = min(ylist)
            link = GaodeLink(url,1)
            link.currentMaxLat = cityMaxLat
            link.currentMaxLon = cityMaxLon
            link.currentMinLat = cityMinLat
            link.currentMinLon = cityMinLon


            # 字段不对
            return [self._build_link(link,1)]
        return None


    def parse(self,content,link):
        page = self._extract_page(content)
        return page,self._extract_link(page,link)
        #  需要判断是否我们需要解析的页面

    def _extract_page(self,content):
        page = GaoDePageInfo(content)
        return page

    def _build_link(self,link,page_num):
        polygon_str = "{},{}|{},{}".format(link.currentMinLat,link.currentMaxLon,link.currentMaxLat,link.currentMinLon)
        # 构建url
        url = 'https://restapi.amap.com/v3/place/polygon?polygon={}&types={}&page={}'.format(polygon_str,self._keywords,page_num)
        _link = GaodeLink(url,page_num)
        _link.currentMaxLat = link.currentMaxLat
        _link.currentMaxLon = link.currentMaxLon
        _link.currentMinLat = link.currentMinLat
        _link.currentMinLon = link.currentMinLon
        return _link

    def _generate_link(self,link):
        currentMinLat = link.currentMinLat
        currentMaxLat = link.currentMaxLat
        currentMaxLon = link.currentMaxLon
        currentMinLon = link.currentMinLon

        links = []
        # 左上
        _link = GaodeLink(None,0)
        _link.currentMinLat = currentMinLat
        _link.currentMaxLon = currentMaxLon
        _link.currentMaxLat = (currentMaxLat + currentMinLat)/2
        _link.currentMinLon = (currentMaxLon + currentMinLon)/2

        links.append(self._build_link(_link, 1))
        # 右上矩形

        _link = GaodeLink(None, 0)
        _link.currentMinLat = (currentMaxLat+currentMinLat)/2
        _link.currentMaxLon = currentMaxLon
        _link.currentMaxLat = currentMaxLat
        _link.currentMinLon = (currentMaxLon+currentMinLon)/2

        links.append(self._build_link(_link, 1))
        # 左下矩形
        _link = GaodeLink(None, 0)
        _link.currentMinLat = currentMinLat
        _link.currentMaxLon = (currentMaxLon+currentMinLon)/2
        _link.currentMaxLat = (currentMaxLat + currentMinLat) / 2
        _link.currentMinLon = currentMinLon
        links.append(self._build_link(_link,1))


        # 右下矩形
        _link = GaodeLink(None, 0)
        _link.currentMinLat = (currentMaxLat+currentMinLat)/2
        _link.currentMaxLon = (currentMaxLon + currentMinLon) / 2
        _link.currentMaxLat = currentMaxLat
        _link.currentMinLon = currentMinLon
        links.append(self._build_link(_link,1))

        return links

    # 执行以下代码，直到count为0的时候跳出循环
    # 生成链接
    def _extract_link(self,page,link):
        extract_links = []
        print(page.count)
        if int(page.count) > 800:
            # 拆分矩形
            _links = self._generate_link(link)
            for _link in _links:
                if not _link.contains(self._crawled):
                    extract_links.append(_link)
        else:
            if int(page.count) == 0:
                return extract_links
            extract_links.append(self._build_link(link,link.page_num + 1))
        return extract_links

    def run(self):
        while not self._crawl_queue.empty():
            link = self._crawl_queue.get()
            content = self.fetch(link)

            page , _links = self.parse(content,link)
            if page is not None:
                self._save(page)
            if page is None:
                continue

            if _links is not None :
                for _link in _links:
                    if _link.contains(self._crawled):
                        continue
                    self._crawl_queue.put(_link)



    def _save(self,page):
        line = page.to_csv_line()
        with open(self._spider_info_file,"a",encoding='utf-8') as f:
            f.write("%s\n" % line)





    def fetch(self,link):
        # 将输进来的矩形进行格式化
        if link is None or link.url is None or link.url in self._crawled:
            return None
        # 用get函数请求数据
        r = requests.get("{}&key={}".format(link.url,self._key.get_key()), headers=self._header)
        # 设置数据的编码为'utf-8'
        r.encoding = 'utf-8'
        # 将请求得到的数据按照'utf-8'编码成字符串
        data = r.text
        print(link.url)
        self._crawled.add(link.url)
        return data

        # 参数处理






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    keys = ["7ca6aef85d9ae336e43f772149f9bf71",
          "71f3467c1bd82a573cbe43ecf72568d5",
          "81a8d050fa7fe43fc11d96b53e9be490",
          "aec401721f5e6c76b923bd434461042a",
          "656246f95a85c57ef09ba95b392b883f",
          "85a9c0545e947656e847e9b4a53b27d5",
          "3974d3dec693d6913269dc670f801b83",
          "7fed391344d67aead6f19f1d923e804f",
          "15b8da74b719ab12ece9589434db16cd",
         ]
    keys = GaodeKey(keys, limit=4500)
    # citys = ["聊城","济宁"]
    citys  = sys.argv[1].split("|")


    for city in citys:
        for keyword in ['家居建材市场',u'物流速递',u'花鸟鱼虫市场',u'农林牧渔基地',u"工厂",u"综合市场",u"公司企业",u"服装鞋帽皮具店"]:
            spider = GaodeSpider(city_name=city,keywords=keyword,tokens=keys)
            spider.run()
