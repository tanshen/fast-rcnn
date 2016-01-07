#Tan Shen

#change xml of gt boxes to rotated images
#coding=utf-8
import  xml.dom.minidom
from xml.dom.minidom import*
import string
#打开xml文档
for line in open("train.txt"):
    line=line.strip('\n')
    #print line
    xmlname="".join([line,'.xml'])
    filename_="".join([line,'_r.jpg'])
    output="".join([line,'_r.xml'])
    #print xmlname
    dom = xml.dom.minidom.parse(xmlname)
    #得到文档元素对象
    root = dom.documentElement

    bb = root.getElementsByTagName('name')
    b= bb[0]
    name0=b.firstChild.data
    #print name0

    bb = root.getElementsByTagName('xmin')
    b= bb[0]
    xmin0=b.firstChild.data
   # print xmin0

    bb = root.getElementsByTagName('ymin')
    b= bb[0]
    ymin0=b.firstChild.data
    #print ymin0

    bb = root.getElementsByTagName('xmax')
    b= bb[0]
    xmax0=b.firstChild.data
   # print xmax0

    bb = root.getElementsByTagName('ymax')
    b= bb[0]
    ymax0=b.firstChild.data
   # print ymax0

    doc = Document()  #创建DOM文档对象

    annotation = doc.createElement('annotation') #创建根元素
    doc.appendChild(annotation)

    folder = doc.createElement('folder')

    folder_text=doc.createTextNode('kaggle')
    annotation.appendChild(folder)
    folder.appendChild(folder_text)

    filename = doc.createElement('filename')
    item = doc.createElement('item')

    item_text=doc.createTextNode(filename_)
    annotation.appendChild(filename)
    filename.appendChild(item)
    item.appendChild(item_text)

    objectname = doc.createElement('object')
    bndbox=doc.createElement('bndbox')
    name = doc.createElement('name')
    xmin=doc.createElement('xmin')
    ymin=doc.createElement('ymin')
    xmax=doc.createElement('xmax')
    ymax=doc.createElement('ymax')

    name_text=doc.createTextNode(name0)
    xmin_=doc.createTextNode(str(ymin0))
    ymin_=doc.createTextNode(str(xmin0))
    xmax_=doc.createTextNode(str(ymax0))
    ymax_=doc.createTextNode(str(xmax0))

    objectname.appendChild(name)
    objectname.appendChild(bndbox)
    xmin.appendChild(xmin_)
    ymin.appendChild(ymin_)
    xmax.appendChild(xmax_)
    ymax.appendChild(ymax_)
    bndbox.appendChild(xmin)
    bndbox.appendChild(ymin)
    bndbox.appendChild(xmax)
    bndbox.appendChild(ymax)
    name.appendChild(name_text)
    annotation.appendChild(objectname)


    ########### 将DOM对象doc写入文件
    outpath="".join(['./result/',output])
    f = open(outpath,'w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()
