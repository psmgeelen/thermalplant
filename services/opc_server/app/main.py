#import RPi.GPIO as GPIO
import random
import asyncio
import sys
import logging

logging.basicConfig(level=logging.WARN, format="%(asctime)s %(name)s %(levelname)s %(message)s")

sys.path.insert(0, "..")

from asyncua import ua, Server
from asyncua.server.history_sql import HistorySQLite

class RandomSensor(object):

    def __init__(self):
        self.name = "RandomSensor"
        self.sources = [
            {
                "name": "random",
                "datatype": ua.Variant(0, ua.VariantType.Double),
                "historize": True,
                "historize_length": 1000,
                "func": self.read_value,
                "writable": True
            },
        ]

    @staticmethod
    def read_value():
        return random.random()

class ThermalSensor(object):

    def __init__(self,  prefix:str, gpio_pin: int):
        self.name = "thermalSensor" + prefix
        self.gpio_pin = gpio_pin
        self.sources = [
            {
                "name": "random",
                "datatype": ua.Variant(0, ua.VariantType.Double),
                "historize": True,
                "historize_length": 1000,
                "func": self.read_value,
                "writable": True
            },
        ]

    #TODO write actual code for SPI read out of thermal sensors.
    @staticmethod
    def read_value():
        return random.random()

async def main():
    # setup our server
    server = Server()

    # Configure server to use sqlite as history database (default is a simple memory dict)
    server.iserver.history_manager.set_storage(HistorySQLite("plantHistorian.sql"))

    # initialize server
    await server.init()

    # TODO template endpoint
    server.set_endpoint("opc.tcp://0.0.0.0:4840/freeopcua/server/")

    # setup our own namespace, not really necessary but should as spec
    uri = "http://examples.freeopcua.github.io"
    idx = await server.register_namespace(uri)

    # get Objects node, this is where we should put our custom stuff
    objects = server.nodes.objects

    devices = []
    devices.append(RandomSensor())
    devices.append(ThermalSensor(prefix="Upper", gpio_pin=10))
    devices.append(ThermalSensor(prefix="Lower", gpio_pin=20))

    # populating our address space
    opcuaComponents = []
    for device in devices:
        opcuaObject =  await  server.nodes.objects.add_object(idx, device.name)

        opcuaVars = []
        for source in device.sources:
            declared_source = {
                "name_device": device.name,
                "name_source": source["name"],
                "datatype": source["datatype"],
                "declaredParam": await opcuaObject.add_variable(idx, source["name"], source["datatype"]),
                "func": source["func"],
                "historize": source["historize"],
                "historize_length": source["historize_length"],
                "writable": source["writable"]
            }

            #
            # if source["writable"]:
            #     await declared_source["declaredParam"].set_writable()

            opcuaVars.append(declared_source)

        opcuaComponents.append(
            {
                "name": device.name,
                "object":opcuaObject,
                "opcuaVars": opcuaVars
            }
        )

    # starting!
    await server.start()

    # enable data change history for this particular node, must be called after start since it uses subscription
    for device in opcuaComponents:
        for source in device["opcuaVars"]:
            if source["historize"]:
                await server.historize_node_data_change(source["declaredParam"], period=None, count=source["historize_length"])

    try:
        count = 0
        while True:
            await asyncio.sleep(1) ## Polling rate
            # Loop to do Measurements
            for component in opcuaComponents:
                for source in component["opcuaVars"]:
                    await source["declaredParam"].set_value(source["func"]())
                    logging.warning(f"Writing value {source["func"]()} from device {source["name_device"]} to {source['name_source']}")

    finally:
        # close connection, remove subscriptions, etc
        await server.stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    loop.run_until_complete(main())