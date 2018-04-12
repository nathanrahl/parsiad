asian: asian.cpp
	$(CXX) -std=c++11 -I/usr/local/include/eigen3/ -O3 asian.cpp -o asian

clean:
	$(RM) asian

.PHONY: clean
