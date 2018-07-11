#N terminal first 10 residues 

filepath='acp.txt'
count=0
countNoOfPeptides=0
countincNo=0
with open(filepath) as file1:
    line=file1.readline()
    with open('first10Cgen.txt','a') as editfile:
        cnt=0
        while line:
            countNoOfPeptides=0
            countincNo+=1
            count=0
            print("hello this shows new line",line)
            for char in line:
                if char.isalpha():
                    countNoOfPeptides+=1
            if countNoOfPeptides>=10:
                newline=line[:10]
                revline=newline[::-1]
                editfile.write(revline+'\n')
            line=file1.readline()
            cnt+=1
        print("countincNo",countincNo)


 with open('ACPSequenceLogo.txt','a') as editfile:#opening a file to write the peptides of length more than 10
        filepath='first10Cgen.txt'
        mystring=" "
        with open(filepath) as file1:       #opening the file that has positive main dataset
                line=file1.readline()
                while line:
                    myString = ">{0}".format(line)
                    myString2=myString.replace ('>', '>\n')
                    editfile.write(myString2) 
                    print("myString",myString)
                    line=file1.readline()
