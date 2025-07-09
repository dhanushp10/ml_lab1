def isvowel(str):
    if(str in ["A","E","I","O","U","a","e","i","o","u"]):
        return True
    else:
        return False

txt=input("Enter a text")
vowel_count=0
consonant_count=0
for i in range(0,len(txt)):
     if(txt[i]!=" "):
       if(isvowel(txt[i])):
           vowel_count= vowel_count+1
       else:
           consonant_count=consonant_count+1    
    
        

print("vowel count is " ,vowel_count)
print("consonant count is",consonant_count)
