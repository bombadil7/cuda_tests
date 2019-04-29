import subprocess

def gen_file(f_name='test.c', num=1):
    prog = r"""
    #include <stdio.h>

    int main(void){
        printf("%f");
        return 3;
    }
    """ % (num)
    with open(f_name, 'w') as f:
        f.write(prog)


if __name__=="__main__":
    f_name = 'test.cu'
    gen_file(f_name, 44.7)
    result = subprocess.run(['/usr/local/cuda/bin/nvcc', 
                        '-o', 
                        'out', 
                        f_name, '--run'], 
                        stdout=subprocess.PIPE)
                        
    total = 1.3 + float(result.stdout)
    print("Total", total)    