import os, sys, re

def check_result(path):
    print( f"Checking {path}..." )
    f = open( path, "r" )
    buffer = []
    for l in f:
        if l.strip() == "": continue
        if "marta_wrapper.h" in l.strip(): continue
        if l.strip().startswith( "#include" ):
            buffer.append(l)
            continue
        if l.strip() == "static inline void kernel(float *restrict p) {": continue
        if l.strip().startswith( "// [" ):
            m = re.search( "\[[^]]*\]", l )
            target_output = list(map(int,m.group().replace("[","").replace("]","").split(",")))
            continue
        if l.strip().startswith( "__m" ):
            if "output" in l.strip():
                output_type = l.strip().split()[0]
                buffer.append( f"static inline {output_type} kernel( float * restrict p ) {{" )
            buffer.append( l )
            continue
        if l.strip().startswith( "DO_NOT_TOUCH" ):
            buffer.append( "return output;" )
            continue

        buffer.append(l)

    with open( "check_kernel.c", "w" ) as output_f:
        for l in buffer:
            output_f.write(l)

        output_f.write( "int main() {" )
        output_f.write( f"float data[{max(target_output)+1}];" )
        output_f.write( f"for( int i = 0; i < {max(target_output)+1}; ++i ) {{" )
        output_f.write( "data[i] = i;}" )
        output_f.write( f"{output_type} output = kernel( data );" )
        for i in range(len(target_output)):
            output_f.write( f"if( output[{i}] != {target_output[-i-1]} ) return {i+1};" )
        output_f.write( "}" )

    if os.system( "gcc -march=native -w check_kernel.c" ) != 0:
        raise ValueError

    ret = os.system( "./a.out" )
    if ret != 0: raise ValueError

    f.close()

if __name__ == "__main__":
    for path in sys.argv[1:]:
        check_result(path)
