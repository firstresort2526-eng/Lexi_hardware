import socket

def get_rpi_ip():
    """Get the RPi's IP address on the WiFi interface"""
    try:
        # Connect to an external server to get the local IP
        # This doesn't actually send data, just gets the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Error getting IP: {e}")
        return None

# Get your RPi's IP
rpi_ip = get_rpi_ip()
print(rpi_ip)