filepath='abcd.txt'
count=0
countNoOfPeptides=0
countincNo=0
with open(filepath) as file1:
    line=file1.readline()
    with open('firdt10.txt','a') as editfile:
        cnt=0
        while line:
            countincNo+=1
            count=0
            print("hello this shows new line",line)
            for char in line:
                if char.isalpha():
                    countNoOfPeptides+=1
            if countNoOfPeptides>=10:
                newline=line[:10]
                editfile.write(newline+'\n')
            line=file1.readline()
            cnt+=1
    print("countincNo",countincNo)
