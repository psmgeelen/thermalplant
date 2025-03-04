import spidev

spi1 = spidev.SpiDev()
spi1.open(1, 0)  # /dev/spidev1.0

spi2 = spidev.SpiDev()
spi2.open(1, 1)  # /dev/spidev1.1

# Configure both devices
spi1.mode = 0b00
spi1.max_speed_hz = 500000

spi2.mode = 0b01
spi2.max_speed_hz = 1000000

# Perform transfers (example data)
response1 = spi1.xfer2([0xAA])
response2 = spi2.xfer2([0xBB])

print(f"SPI 1.0 Response: {response1}")
print(f"SPI 1.1 Response: {response2}")

# Close devices
spi1.close()
spi2.close()
