debug:
	cmake -DCMAKE_BUILD_TYPE=Debug -Btarget && cd target && make

release:
	cmake -DCMAKE_BUILD_TYPE=Release -Btarget && cd target && make

run_debug: debug
	./target/Debug/SpikingCpp

run_release: release
	./target/Release/SpikingCpp