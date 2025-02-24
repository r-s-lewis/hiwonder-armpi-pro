#!/usr/bin/env python3
import time
import serial
import pigpio
import ctypes


LOBOT_SERVO_FRAME_HEADER         = 0x55
LOBOT_SERVO_MOVE_TIME_WRITE      = 1
LOBOT_SERVO_MOVE_TIME_READ       = 2
LOBOT_SERVO_MOVE_TIME_WAIT_WRITE = 7
LOBOT_SERVO_MOVE_TIME_WAIT_READ  = 8
LOBOT_SERVO_MOVE_START           = 11
LOBOT_SERVO_MOVE_STOP            = 12
LOBOT_SERVO_ID_WRITE             = 13
LOBOT_SERVO_ID_READ              = 14
LOBOT_SERVO_ANGLE_OFFSET_ADJUST  = 17
LOBOT_SERVO_ANGLE_OFFSET_WRITE   = 18
LOBOT_SERVO_ANGLE_OFFSET_READ    = 19
LOBOT_SERVO_ANGLE_LIMIT_WRITE    = 20
LOBOT_SERVO_ANGLE_LIMIT_READ     = 21
LOBOT_SERVO_VIN_LIMIT_WRITE      = 22
LOBOT_SERVO_VIN_LIMIT_READ       = 23
LOBOT_SERVO_TEMP_MAX_LIMIT_WRITE = 24
LOBOT_SERVO_TEMP_MAX_LIMIT_READ  = 25
LOBOT_SERVO_TEMP_READ            = 26
LOBOT_SERVO_VIN_READ             = 27
LOBOT_SERVO_POS_READ             = 28
LOBOT_SERVO_OR_MOTOR_MODE_WRITE  = 29
LOBOT_SERVO_OR_MOTOR_MODE_READ   = 30
LOBOT_SERVO_LOAD_OR_UNLOAD_WRITE = 31
LOBOT_SERVO_LOAD_OR_UNLOAD_READ  = 32
LOBOT_SERVO_LED_CTRL_WRITE       = 33
LOBOT_SERVO_LED_CTRL_READ        = 34
LOBOT_SERVO_LED_ERROR_WRITE      = 35
LOBOT_SERVO_LED_ERROR_READ       = 36



class ServoBusController:

    def __init__(self, serial_port="/dev/ttyS0", baud_rate=115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.pi = pigpio.pi()
        self.rx_pin = 4
        self.tx_pin = 27
        self.serial_handle = self._init_serial()

        self._init_gpio()


    def _init_gpio(self):
        self.pi.set_mode(self.rx_pin, pigpio.OUTPUT)
        self.pi.write(self.rx_pin, 0)
        self.pi.set_mode(self.tx_pin, pigpio.OUTPUT)
        self.pi.write(self.tx_pin, 1)

    def _port_read(self):  # 配置单线串口为输入
        self.pi.write(self.rx_pin, 1)  # 拉高RX_CON 即 GPIO17
        self.pi.write(self.tx_pin, 0)  # 拉低TX_CON 即 GPIO27
        

    def _port_write(self):  # 配置单线串口为输出
        self.pi.write(self.tx_pin, 1)  # 拉高TX_CON 即 GPIO27
        self.pi.write(self.rx_pin, 0)  # 拉低RX_CON 即 GPIO17


    def _init_serial(self):
        try:
            return serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        except serial.SerialException as e:
            print(f"[ERROR] Serial connection failed: {e}")
            return None


    def _reconnect_serial(self):
        print(f'[DEBUG] Reconnecting the serial connection...')
        if self.serial_handle and self.serial_handle.is_open:
            self.serial_handle.close()
        time.sleep(0.2)
        self.serial_handle = self._init_serial()


    def _checksum(self, buf):
        return (~sum(buf[2:]) & 0xFF)


    def _read_response(self, expected_cmd, response_length=7):
        """Reads and validates a response from the serial bus."""
        if self.serial_handle is None or not self.serial_handle.is_open:
            print("[ERROR] Serial connection lost, attempting reconnection...")
            self._reconnect_serial()
            return None

        try:
            self.serial_handle.flushInput()
            time.sleep(0.005)  # Give time for the response

            count = self.serial_handle.inWaiting()
            if count == 0:
                print("[WARNING] No data received.")
                return None

            response = self.serial_handle.read(count)

            if len(response) < response_length:
                print(f"[WARNING] Incomplete response received: {response}")
                return None

            if response[0] == LOBOT_SERVO_FRAME_HEADER and response[1] == LOBOT_SERVO_FRAME_HEADER:
                if response[4] == expected_cmd:
                    dat_len = response[3]
                    if dat_len == 4:
                        return response[5]
                    elif dat_len == 5:
                        return ctypes.c_int16(response[5] | (response[6] << 8)).value
                    elif dat_len == 7:
                        pos1 = ctypes.c_int16(response[5] | (response[6] << 8)).value
                        pos2 = ctypes.c_int16(response[7] | (response[8] << 8)).value
                        return pos1, pos2
            else:
                print(f"[WARNING] Invalid response header: {response}")

        except serial.SerialException as e:
            print(f"[ERROR] Serial Read Error: {e}")
            self._reconnect_serial()

        return None  # Return None if no valid response
    

    def send_command(self, servo_id, command, dat1=None, dat2=None):
        if self.serial_handle is None or not self.serial_handle.is_open:
            print("[ERROR] Serial connection lost, attempting reconnection...")
            self._reconnect_serial()
            return
    
        try:
            self._port_write()
            buf = bytearray(b'\x55\x55')  # 帧头
            buf.append(servo_id)
            # 指令长度
            if dat1 is None and dat2 is None:
                buf.append(3)
            elif dat1 is not None and dat2 is None:
                buf.append(4)
            elif dat1 is not None and dat2 is not None:
                buf.append(7)

            buf.append(command)  # 指令
            # 写数据
            if dat1 is None and dat2 is None:
                pass
            elif dat1 is not None and dat2 is None:
                buf.append(dat1 & 0xff)  # 偏差
            elif dat1 is not None and dat2 is not None:
                buf.extend([(0xff & dat1), (0xff & (dat1 >> 8))])  # 分低8位 高8位 放入缓存
                buf.extend([(0xff & dat2), (0xff & (dat2 >> 8))])  # 分低8位 高8位 放入缓存
            # 校验和
            buf.append(self._checksum(buf))
            self.serial_handle.write(buf)  # 发送
        
        except serial.SerialException as e:
            print(f"Serial Write Error: {e}")
            self._reconnect_serial()


    def move_servo(self, servo_id, position, duration=250):
        position = max(0, min(position, 1000))
        duration = max(0, min(duration, 30000))


        
        self.send_command(servo_id, LOBOT_SERVO_MOVE_TIME_WRITE, position, duration)


    def stop_servo(self, servo_id):
        self.send_command(servo_id, LOBOT_SERVO_MOVE_STOP)


    def get_servo_position(self, servo_id, max_attempts=5):
        """
        Gets the current position of the servo with a retry mechanism.
        """
        for attempt in range(max_attempts):
            self.serial_servo_read_cmd(servo_id, LOBOT_SERVO_POS_READ)
            msg = self.serial_servo_get_rmsg(LOBOT_SERVO_POS_READ)

            if msg is not None:
                return msg

            # print(f"[WARNING] No response from servo {servo_id}. Retrying {attempt + 1}/{max_attempts}...")
            # time.sleep(0.05)

        # print(f"[ERROR] Failed to read position from servo {servo_id}.")
        return None  # Return None if all attempts fail


    
    def serial_servo_get_rmsg(self, cmd):
        """
        Reads serial response from the servo, ensuring valid data.
        """
        if self.serial_handle is None or not self.serial_handle.is_open:
            print("Error: Serial connection lost. Reconnecting...")
            self._reconnect_serial()
            return None

        try:
            self.serial_handle.flushInput()
            self._port_read()
            time.sleep(0.005)
            count = self.serial_handle.inWaiting()

            if count == 0:
                return None  # No data received

            recv_data = self.serial_handle.read(count)
            if len(recv_data) < 7:  # Minimum expected packet length
                # print("Warning: Incomplete data received", recv_data)
                return None

            if recv_data[0] == 0x55 and recv_data[1] == 0x55 and recv_data[4] == cmd:
                dat_len = recv_data[3]
                if dat_len == 4:
                    return recv_data[5]
                elif dat_len == 5:
                    return ctypes.c_int16(recv_data[5] | (recv_data[6] << 8)).value
                elif dat_len == 7:
                    pos1 = ctypes.c_int16(recv_data[5] | (recv_data[6] << 8)).value
                    pos2 = ctypes.c_int16(recv_data[7] | (recv_data[8] << 8)).value
                    return pos1, pos2
            else:
                return None
        
        except serial.SerialException as e:
            print(f"Serial Read Error: {e}")
            self._reconnect_serial()
        
        return None
            

    def serial_servo_read_cmd(self, id=None, r_cmd=None):
        self._port_write()
        buf = bytearray(b'\x55\x55')  # 帧头
        buf.append(id)
        buf.append(3)  # 指令长度
        buf.append(r_cmd)  # 指令
        buf.append(self._checksum(buf))  # 校验和
        self.serial_handle.write(buf)  # 发送
        time.sleep(0.00034)


    def close(self):
        if self.serial_handle and self.serial_handle.is_open:
            self.serial_handle.close()