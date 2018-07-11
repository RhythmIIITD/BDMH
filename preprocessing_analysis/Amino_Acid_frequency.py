#Code snippet for finding the average whole amino acid composition of Non-ACPs.
#(The same code was used to get the whole amino acid composition of ACPs and
#AMPs by changing the dataset file that was opened using the file handle.
from collections import Counter
countpep=0
#Total count gives the count of the number of peptides in one sequence.
total_count=0
#Opening the file having Non_ACP dataset
with open('non_antiCancerPeptides.txt') as file_handler:
    lines = file_handler.read().splitlines()
for line in lines:
	for char in line:
		total_count+=1
#intializing the intial count of each peptide to 0(zero).
countA = 0
countC = 0
countD = 0
countE = 0
countF = 0
countG = 0
countH = 0
countI = 0
countK = 0
countL = 0
countM = 0
countN = 0
countP = 0
countQ = 0
countR = 0
countS = 0
countT = 0
countV = 0
countW = 0
countY = 0
#each count value gets updated by the following snippet
for amino in lines:
     amino_list = list(amino)
     for ele in amino_list:
        if(ele == 'A'):
            countA+=1
        elif(ele == 'C'):
            countC+=1
        elif (ele == 'D'):
            countD += 1
        elif (ele == 'E'):
            countE += 1
        elif (ele == 'F'):
            countF += 1
        elif (ele == 'G'):
            countG += 1
        elif (ele == 'H'):
            countH += 1
        elif (ele == 'I'):
            countI += 1
        elif (ele == 'K'):
            countK += 1
        elif (ele == 'L'):
            countL += 1
        elif (ele == 'M'):
            countM += 1
        elif (ele == 'N'):
            countN += 1
        elif (ele == 'P'):
            countP += 1
        elif (ele == 'Q'):
            countQ += 1
        elif (ele == 'R'):
            countR += 1
        elif (ele == 'S'):
            countS += 1
        elif (ele == 'T'):
            countT += 1
        elif (ele == 'V'):
            countV += 1
        elif (ele == 'W'):
            countW += 1
        elif (ele == 'Y'):
            countY += 1
#calculating the whole average amino acid composition of each peptide 
percentageA = (countA/total_count) * 100
percentageC = (countC/total_count) * 100
percentageD = (countD/total_count) * 100
percentageE = (countE/total_count) * 100
percentageF = (countF/total_count) * 100
percentageG = (countG/total_count) * 100
percentageH = (countH/total_count) * 100
percentageI = (countI/total_count) * 100
percentageK = (countK/total_count) * 100
percentageL = (countL/total_count) * 100
percentageM = (countM/total_count) * 100
percentageN = (countN/total_count) * 100
percentageP = (countP/total_count) * 100
percentageQ = (countQ/total_count) * 100
percentageR = (countR/total_count) * 100
percentageS = (countS/total_count) * 100
percentageT = (countT/total_count) * 100
percentageV = (countV/total_count) * 100
percentageW = (countW/total_count) * 100
percentageY = (countY/total_count) * 100
# printing the calculated value of each peptide composition in percentage
print (percentageA)
print (percentageC)
print (percentageD)
print (percentageE)
print (percentageF)
print (percentageG)
print (percentageH)
print (percentageI)
print (percentageK)
print (percentageL)
print (percentageM)
print (percentageN)
print (percentageP)
print (percentageQ)
print (percentageR)
print (percentageS)
print (percentageT)
print (percentageV)
print (percentageW)
print (percentageY)







