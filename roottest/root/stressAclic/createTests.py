for index in range(200):
    code = '''
class Class%s{};
int test%s(){return 0;}
''' %(index,index)
    with open(f'test{index}.C','w') as f:
        f.write(code)