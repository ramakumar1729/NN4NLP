import xml.etree.ElementTree as ET


def getOutputs(textID):
    f= open('trainY.txt', 'r')
    fOuts=[]
    output= f.readlines()
    for out in output:
        doc= out[out.index('(')+1:(out.index('.'))]
        if doc== textID:
            out=out[:-1]
            items= out.split(',')
            if len(items)==2:
                ent1= items[0][items[0].index('.')+1:]
                ent2= items[1][items[1].index('.')+1:-1]
            else:
                ent2 = items[0][items[0].index('.') + 1:]
                ent1 = items[1][items[1].index('.') + 1:]
            rel= [out[:out.index('(')], 'entity'+ ent1, 'entity'+ent2]
            fOuts.append(rel)
    return fOuts

def getOutputs2(textID):
    f= open('trainY.txt', 'r')
    fOuts=[]
    output= f.readlines()
    for out in output:
        doc= out[out.index('(')+1:(out.index('.'))]
        if doc== textID:
            out=out[:-1]
            items= out.split(',')
            if len(items)==2:
                ent1= items[0][items[0].index('.')+1:]
                ent2= items[1][items[1].index('.')+1:-1]
            else:
                ent2 = items[0][items[0].index('.') + 1:]
                ent1 = items[1][items[1].index('.') + 1:]
            rel= [out[:out.index('(')], 'T'+ ent1, 'T'+ent2]
            fOuts.append(rel)
    return fOuts

def addEntities(entity1, entity2, line):
    index1= line.index(entity1)+ len(entity1)+1
    index2= line.index('</'+entity1)
    line2= line[:index1]+ '<P>'+ line[index1:index2]+ '</P>'+ line[index2:]
    index1 = line2.index(entity2) + len(entity2) + 1
    index2 = line2.index('</' + entity2)
    fLine= line2[:index1]+ '<C>'+ line2[index1:index2]+ '</C>'+ line2[index2:]
    return fLine

def constructOutput(ent1, ent2, outputs):
    for out in outputs:
        if out[1]== ent1 and out[2]==ent2:
            output= out[0]+ '('+ ent1 + ','+ ent2+ ')'
            return output
    output= 'NONE('+ ent1+ ','+ ent2+')'
    return output


def getCandidates(size):
    list=[]
    for p in range(1, size+1):
        for q in range(p-3, p+4):
            if q>0 and q< size+1 and p!=q:
                list.append([p, q])
    return list

# fTrainY= open('finaltrainY.txt', 'w')
# fTrainX= open('finaltrainX.txt', 'w')
tree= ET.parse('test.text.xml')
root= tree.getroot()

# for doc in root.findall('text'):
#     id= doc.get('id')
#     outputs= getOutputs(id)
#     abs= doc.find('abstract')
#     absIter = abs.itertext()
#     run= True
#     ent = False
#     entity=1
#     fLine=""
#     while run:
#         try:
#             text= absIter.next()
#             if ent:
#                 fLine= fLine+ '<entity' + str(entity) + '>'
#                 fLine+= text
#                 fLine+= '</entity' + str(entity) + '>'
#                 entity += 1
#                 ent = False
#             else:
#                 ent=True
#                 newT=text
#                 for i in range(len(text)):
#                     if ord(text[i])>128:
#                         newT= newT[:i]+ newT[i+1:]
#                 fLine+= str(newT)
#         except StopIteration:
#             run= False
#     candidates= getCandidates(entity-1)
#     for (ent1, ent2) in candidates:
#         newLine= addEntities('entity'+str(ent1), 'entity'+str(ent2), fLine)
#         newY= constructOutput('entity'+str(ent1), 'entity'+str(ent2), outputs)
#         newLine=newLine.replace('\n', '')
#         fTrainX.write(newLine)
#         fTrainX.write('\n')
#         fTrainY.write(newY)
#         fTrainY.write('\n')
# fTrainX.close()
# fTrainY.close()

for doc in root.findall('text'):
    id= doc.get('id')
    outputs= getOutputs2(id)
    abs= doc.find('abstract')
    absIter = abs.itertext()
    run= True
    ent = False
    entity=1
    currText=""
    index=0
    entFile= open('Test2018/'+ id+ '.ann', 'w')
    textFile = open('Test2018/' + id + '.txt', 'w')
    while run:
        try:
            text= absIter.next()
            if ent:
                cleanText=""
                for l in text:
                    if ord(l)>128:
                        cleanText+= "'"
                    else:
                        cleanText+= l
                text= cleanText
                currText += text
                entFile.write('T'+ str(entity)+ '\t'+ 'NoType '+str(index)+' '+str(index+len(text)) +'\t'+ text+ '\n')
                ent = False
                entity += 1
                index+= len(text)
            else:
                ent=True
                newT=text
                for i in range(len(text)):
                    if ord(text[i])>128:
                        newT= newT[:i]+ newT[i+1:]
                currText += newT
                index += len(newT)
        except StopIteration:
            run= False
    textFile.write(currText+ '\n')
    textFile.close()
    relIndex=1
    for relation in outputs:
        rel= relation[0]
        ent1= relation[1]
        ent2= relation[2]
        entFile.write('R'+ str(relIndex)+'\t'+rel+" Arg1:"+ ent1+" Arg2:"+ ent2 +'\n')
        relIndex+=1
    entFile.close()
