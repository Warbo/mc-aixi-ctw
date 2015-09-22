aixi: src/main.o src/agent.o src/search.o src/predict.o src/environment.o src/util.o src/stdenv.o
	g++ -O3 -Wall -ggdb -o aixi src/*.o

clean:
	rm -f aixi src/*.o
