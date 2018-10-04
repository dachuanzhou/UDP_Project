SUBDIRS= UDP_Rcv txt2png

.PHONY:default all clean $(SUBDIRS)

default:all

all clean:

#  this 'make' cmd will cause the Makefile be read again by 'make'
	$(MAKE) $(SUBDIRS) TARGET=$@

$(SUBDIRS):print
	@echo "--- TARGET=$(TARGET)"
	$(MAKE) -C $@ $(TARGET)

print:
	@echo "--- MAKE  =$(MAKE)"
	@echo "--- TARGET=$(TARGET)"

