def tilt_hareket(value) -> str:
    """
    Verilen tamsayı değerini hexadecimal (onaltılık) kod biçimine dönüştürüp,
    negatif değerler için 2's complement yöntemiyle formatlar.

    :param value: Dönüştürülecek tamsayı değeri
    :return: Biçimlendirilmiş hexadecimal string (örnek: '0xFE/0xC7/0x80')
    """
    if value:
        if value < 0:
            # Negatif değerler için 2's complement hesaplama
            value = (1 << 24) + value  # 24-bitlik bir değer
        # Hexadecimal formatta 6 haneli string
        #print(f'deneme{value}')
        #hex_value = f"{value:06X}"
        hex_value=value.to_bytes(3,byteorder='big')
        formatted_hex = b'\x55\x01\x02\x20\x04\x02' + hex_value

        formatted_hex = ''.join(f'\\x{byte:02x}' for byte in formatted_hex)
        #formatted_hex = pan + formatted_hex

        return formatted_hex




#value = -80000
#formatted_hex_code = tilt_hareket(value)
#print(formatted_hex_code)