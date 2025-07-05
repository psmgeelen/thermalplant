import asyncio
import sys
import logging
from asyncua import ua, Server
from asyncua.server.history_sql import HistorySQLite
from sensors import RandomSensor, ThermalSensor, RPMSensor, AudioSensorMfcc, AudioSensorSpectral, VoltageSensor
import time
logging.basicConfig(level=logging.WARN, format="%(asctime)s %(name)s %(levelname)s %(message)s")

sys.path.insert(0, "..")

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
    devices.append(ThermalSensor(prefix="Upper", base_url="http://dmz-sensors/temperature_upper"))
    devices.append(ThermalSensor(prefix="Lower", base_url="http://dmz-sensors/temperature_lower"))
    devices.append(RPMSensor(base_url="http://dmz-sensors/rpm"))
    devices.append(VoltageSensor(base_url="http://dmz-sensors/voltage"))
    devices.append(AudioSensorMfcc(base_url="http://dmz-sensors/mfcc"))
    devices.append(AudioSensorSpectral(base_url="http://dmz-sensors/spectrum"))

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
                "declaredParam": await opcuaObject.add_variable(idx, source["name"], source["datatype"].Value),
                "initial_value": source["datatype"].Value,
                "func": source["func"],
                "historize": source["historize"],
                "NodeId": "",
                "DataType": source["datatype"].VariantType,
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
                source["NodeId"] = source["declaredParam"].nodeid
                logging.warning(f"Historized NodeId {source['NodeId']} with DataType {source['DataType']}")

    try:
        count = 0
        while True:
            # await asyncio.sleep(1) ## Polling rate
            # Loop to do Measurements
            for component in opcuaComponents:
                logging.warning(f"Processing device {component['name']}")
                for source in component["opcuaVars"]:
                    new_value = source["func"]()
                    await asyncio.sleep(0.4) ## Polling rate
                    await source["declaredParam"].set_value(new_value)
                    logging.warning(f"Writing value {new_value} to variable {source['name_source']} under device {source['name_device']} with NodeId {source['NodeId']}")

    finally:
        # close connection, remove subscriptions, etc
        await server.stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    loop.run_until_complete(main())