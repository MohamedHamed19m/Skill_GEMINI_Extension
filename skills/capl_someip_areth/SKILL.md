---
name: capl-someip-areth
description: Expert for generating, analyzing, and debugging CAPL code using the AUTOSAR Ethernet Interaction Layer (AREth) functions for SOME/IP communication. Use this skill when the user asks to (1) create or modify CAPL code for SOME/IP/Ethernet, (2) implement SOME/IP Service Discovery (SD) logic, (3) write event, method, or field handlers, or (4) utilize specific AREth functions like AREthCreateMethodCall, AREthSendEvent, or OnAREthMethodRequest.
keywords: ["areth", "someip", "capl"]
---
# CAPL for AUTOSAR Ethernet Interaction Layer (SOME/IP)

## Overview

This skill enables generation of production-grade CAPL code for SOME/IP communications in CANoe, adhering strictly to AUTOSAR Ethernet IL APIs. CAPL scripts here simulate ECU behaviors for service-oriented Ethernet in automotive systems: consuming/providing services, handling methods/events/fields, service discovery (SD), TCP/TLS connections, and raw message manipulation.

**Core Principles**:
- **AUTOSAR Compliance**: All code uses official `AREth*` functions (e.g., `AREthCreateConsumedServiceInstance`, `AREthTriggerEvent`).
- **Error Handling**: Always check `AREthGetLastError()` and log with `write()`.
- **Lifecycle**: Initialize with `AREthILControlInit()`, start/stop with `AREthILControlStart/Stop`.
- **Testing Focus**: Include setup/teardown, symbolic DB integration where possible.
- **Efficiency**: Use callbacks for async events; avoid polling.

**Prerequisites**:
- CANoe DE or CANoe4SW DE with Ethernet option.
- ARXML or DBC with SOME/IP definitions loaded.
- CAPL environment: Include `<Eth.h>`, `<AREthIL.h>` (assumed available).

**Success Criteria**:
- Code compiles without warnings.
- Handles normal/edge cases (e.g., timeouts, invalid responses).
- Outputs: Structured CAPL with comments, includes test stubs.
- Validation: Suggest `elapse(100)` delays for async; log all API calls.

## Workflow Decision Tree

Route based on user request type:

### Client-Side (Consuming Services/Events/Methods)
Use when request involves subscribing, calling methods, or receiving notifications.
- **Service Discovery**: Start with `AREthSDRequireService()`.
- **Event/Field Subscription**: `AREthAddConsumedEventGroup()`, then `AREthCreateEventConsumer()` or `AREthCreateFieldConsumer()`.
- **Method Calls**: `AREthCreateMethodCall()` with `OnAREthMethodResponse` callback, then `AREthCallMethod()`.
- **Release**: `AREthReleaseConsumedServiceInstance()`.

### Server-Side (Providing Services/Events/Methods)
Use for offering services, handling requests, triggering events.
- **Setup**: `AREthCreateProvidedServiceInstance()`, add methods/events/fields via `AREthAddMethod()`, `AREthAddEvent()`.
- **Event Groups**: `AREthAddProvidedEventGroup()`, assign with `AREthAddEventToEventgroup()`.
- **Handle Requests**: Implement `OnAREthMethodRequest` callback for responses.
- **Trigger Events**: `AREthTriggerEvent()` or `AREthCommitField()`.
- **SD Offer**: `AREthSDSetServiceStatus()`.

### General/Control
- **Init/Control**: Always wrap in `on start`/`on stop` with `AREthILControlInit()`, `AREthILControlStart()`.
- **Endpoints/TCP/TLS**: `AREthOpenLocalApplicationEndpoint()`; for TCP: `AREthEstablishTCPConnection()`; TLS: `AREthTlsAuthenticateAs*()`.
- **Raw/Low-Level**: `AREthCreateMessage()`, `AREthSetData()`, `AREthOutputMessage()` for custom PDUs.
- **Value Access**: Use `AREthSetValue*()`/`AREthGetValue*()` for params (e.g., `AREthSetValueDWord(handle, paramId, value)`).

### Raw Data/Message Processing
- For custom serialization: `AREthSerializeMessage()`, `AREthGetData()`.
- Callbacks: `OnAREthMessage()` for RX/TX processing.

**Edge Cases**:
- Errors: Check `AREthGetLastError()` after each call; use `OnAREthMethodError` for methods.
- IPv6/TLS: Specify in endpoints; handle `OnAREthEstablishedTLSConnection`.
- Multicast: `AREthSetMulticastReceiverEndpoints()` for events.

## Execution Steps

1. **Parse Request**: Identify client/server, key APIs (e.g., method call → `AREthCallMethod`).
2. **Structure CAPL**:
   - `variables { ... }` for handles (e.g., `dword serviceHandle;`).
   - `on start { AREthILControlInit(); AREthILControlStart(); }`.
   - Implement callbacks (e.g., `on message SomeIPMsg { ... }` or custom `OnAREth*`).
   - `on stop { AREthILControlStop(); }`.
3. **Integrate Symbolic DB**: Use `AREthGetConsumedObjectHandle()` for dynamic lookup.
4. **Add Logging/Validation**: `write("Error: %d", AREthGetLastError());`.
5. **Test Stub**: Include `testmodule` with `testcase` for normal/edge scenarios.

## Examples

### Example 1: Client-Side Event Subscription (Normal Case)
User: "CAPL to subscribe to SOME/IP event group and log notifications."

```capl
variables {
  dword consumedServiceHandle;
  dword eventGroupHandle;
  dword eventConsumerHandle;
}

on start {
  // Create consumed service instance (ServiceID=0x1234, InstanceID=0x0001)
  consumedServiceHandle = AREthCreateConsumedServiceInstance(0x1234, 0x0001, this);
  if (AREthGetLastError() != 0) write("Error creating service: %d", AREthGetLastError());

  // Add event group (MajorVersion=1)
  eventGroupHandle = AREthAddConsumedEventGroup(consumedServiceHandle, 0x5678, 1);  // EventGroupID=0x5678

  // Create event consumer (EventID=0x9ABC)
  eventConsumerHandle = AREthCreateEventConsumer(consumedServiceHandle, 0x9ABC, this);
  if (AREthGetLastError() != 0) write("Error creating consumer: %d", AREthGetLastError());

  // Require service via SD
  AREthSDRequireService(consumedServiceHandle, 0xDEADBEEF, 30490);  // IPv4 dest, port

  AREthILControlStart();
}

on key 's' {  // Simulate stop
  AREthSDReleaseService(consumedServiceHandle);
  AREthRemoveEventConsumer(consumedServiceHandle, eventConsumerHandle);
  AREthRemoveConsumedEventGroup(consumedServiceHandle, eventGroupHandle);
  AREthReleaseConsumedServiceInstance(consumedServiceHandle);
  AREthILControlStop();
}

// Callback for notifications
OnAREthFieldNotification(dword handle, dword notifierId) {
  dword value;
  AREthGetValueDWord(handle, 1, value);  // Param 1
  write("Event notified: Value = %d", value);
}
Why Strong: Initializes lifecycle, handles errors, uses SD, includes teardown. Edge: Add timeout check in callback.
Example 2: Server-Side Method Handler (Edge Case with Error)
User: "CAPL for server method response, handling invalid requests."
caplvariables {
  dword providedServiceHandle;
  dword methodHandle;
}

on start {
  providedServiceHandle = AREthCreateProvidedServiceInstance(0x1234, 0x0001, this);  // Service/Instance
  methodHandle = AREthAddMethod(providedServiceHandle, 0x5678, 1);  // MethodID, Version
  AREthSDSetServiceStatus(providedServiceHandle, 1);  // Available
  AREthILControlStart();
}

OnAREthMethodRequest(dword handle, dword sessionHandle, dword methodId) {
  if (methodId != 0x5678) {
    // Error response
    AREthSetReturnCode(handle, 0x03);  // E_NOT_OK
    return;
  }
  // Set response values
  dword responseValue = 42;
  AREthSetValueDWord(handle, 1, responseValue);
  AREthSetReturnCode(handle, 0x00);  // OK
}

on stop {
  AREthRemoveMethod(providedServiceHandle, methodHandle);
  AREthReleaseProvidedServiceInstance(providedServiceHandle);
  AREthILControlStop();
}
Why Strong: Implements callback, validates input, sets return code. Edge: Handles OnAREthMethodError