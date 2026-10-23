import pymcprotocol

plc = pymcprotocol.Type3E()

try:
    # Connect to PLC
    plc.connect("192.168.3.250",12288)

    # Read D200
    value = plc.batchread_wordunits(headdevice="D200", readsize=1)

    print("D200 =", value)

except Exception as e:
    print("Error:", e)

finally:
    # Close connection if it was opened
    try:
        plc.close()
    except:
        pass