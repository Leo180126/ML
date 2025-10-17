# Đường dẫn đến file test
test_file = tests/test_read_csv.c

# Mục tiêu mặc định
all: test.a run

# Biên dịch chương trình
test.a: src/*.c $(test_file)
	gcc -Iinclude src/*.c $(test_file) -o test.a -lm

# Chạy chương trình
run:
	./test.a

# Biên dịch với debug symbols để dùng gdb
debug: src/*.c $(test_file)
	gcc -g -Iinclude src/*.c $(test_file) -o test.a -lm
	gdb test.a

# Dọn dẹp file biên dịch
clean:
	rm -f test.a
	clear
