import pan
import codecs

def format_data_with_crc(data) -> str:
    """
    Verilen veriye CRC-16/XMODEM hesaplayarak verinin sonuna ekler ve
    hexadecimal formatta bir string olarak döndürür.

    :param data: Verilen veri dizisi (bytes)
    :return: CRC'li ve formatlanmış veri stringi
    """

    # CRC hesaplama fonksiyonu
    def crc16_xmodem(data: bytes) -> int:
        crc = 0x0000
        polynomial = 0x1021

        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ polynomial
                else:
                    crc <<= 1
                crc &= 0xFFFF  # 16-bit sınırını koru

        return crc

    # CRC değerini hesapla
    crc_value = crc16_xmodem(data)


    # CRC değerini bayt dizisi olarak elde et
    crc_bytes = crc_value.to_bytes(2, byteorder='big')

    # CRC değerini verinin sonuna ekle
    data_with_crc = data + crc_bytes

    # Veriyi ve CRC'yi hexadecimal formatta birleştir
    formatted_data_with_crc = ''.join(f'\\x{byte:02x}' for byte in data_with_crc)
    formatted_data_with_crc = codecs.decode(formatted_data_with_crc.replace(r'\x', ''), 'hex')

    return formatted_data_with_crc


# Verilen veri dizisi
#data = b'\x55\x01\x02\x1c\x07\x10\x00\xc3\x50\x00\x03\xe8'
#byte = codecs.decode(pan.formatted_hex_code.replace(r'\x', ''), 'hex')

# Fonksiyonu çağırarak CRC'li veriyi formatla
#formatted_data = format_data_with_crc(byte)


#print(f"CRC'li veri: {formatted_data}")
#print(type(formatted_data))
