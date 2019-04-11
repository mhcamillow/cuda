############################# Makefile ##########################
all: fileUtils.o main.o
fileUtils.o: fileUtils.cpp
	g++ -c fileUtils.cpp -o bin/fileUtils.o
main.o: main.cu
	nvcc bin/fileUtils.o main.cu -o bin/main
	sudo chmod +x bin/main
	./bin/main train-m.txt test-m.txt 3
clean:
	rm -rf bin/*