data = open("input2.txt", 'r')
flag = 1
flag2 = 1
processed_data = ""
for index, line in enumerate(data):
    if "_pp(dict_data):" in line:
        assert  1 == 1
    if index == 100000:
        break
    
    print(f"The processed line is {index}")
    if '"""' in line:
        flag = flag * -1
        continue
    
    if "'''" in line:
        flag2 = flag2 * -1
        
    if flag>0 and flag2>0:
        if 'def' in line:
            line = "\n" + line
            assert 1 == 1
            
        processed_data = processed_data + line
        
with open('code_data.txt', 'w') as f:
    f.write(processed_data)
    
