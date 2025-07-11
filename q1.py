#to check number of vowels and consonants

def isvowel(str):
    if(str in ["A","E","I","O","U","a","e","i","o","u"]):
        return True
    else:
        return False
#checks if vowel is present or not 


txt=input("Enter a text")
vowel_count=0
consonant_count=0

#if vowel exist it will be checked if it is a space it will skip
for i in range(0,len(txt)):
     if(txt[i]!=" "):
       if(isvowel(txt[i])):
           vowel_count= vowel_count+1
       else:
           consonant_count=consonant_count+1    
    
        

print("vowel count is " ,vowel_count)
print("consonant count is",consonant_count)
