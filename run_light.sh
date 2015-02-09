#!/bin/sh
make && \
stty -F /dev/ttyACM0 sane raw pass8 -echo -hupcl clocal 9600 && \
sleep 2 && \
./aixi conf/light_sensor.conf light_logs/1_bit.log
