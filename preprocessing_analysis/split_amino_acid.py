filepath='non_acp.txt'
count=0
countincOfLine=0
with open(filepath) as file1:
    line=file1.readline()
    with open('non_acp1.txt','a') as editfile:
        cnt=0
        with open('non_acp2.txt','a') as editfile1:
            while line:
                countincOfLine+=1
                count=0
                print("hello",line)
                for char in line:
                    count+=1
                    counthalf1=count/2
                    counthalf=int(counthalf1)
                    print("counthalf",counthalf)
                    firsthalf=line[:-counthalf]
                    print("firsthalf here",firsthalf)
                    lasthalf=line[-counthalf:]
                    print("lasthalf here ",lasthalf)
                    editfile.write(firsthalf+'\n')
                    editfile1.write(lasthalf)    
                line=file1.readline()
                cnt+=1
            print("countincOfLine",countincOfLine)     
