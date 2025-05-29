import { describe, expect, it } from "vitest";
import { processPartialToolCalls, processToolCalls } from "../src/utils";

describe("processPartialToolCalls", () => {
	it("should merge partial tool calls by index", () => {
		const partialCalls = [
			{
				index: 0,
				id: "call_123",
				type: "function",
				function: { name: "test_func", arguments: '{"par' },
			},
			{
				index: 0,
				function: { arguments: 'am": "val' },
			},
			{
				index: 0,
				function: { arguments: 'ue"}' },
			},
		];
		const result = processPartialToolCalls(partialCalls);
		expect(result).toHaveLength(1);
		expect(result[0].args).toBe('{"param": "value"}');
		expect(result[0].toolName).toBe("test_func");
		expect(result[0].toolCallId).toBe("call_123");
	});

	it("should handle multiple partial tool calls with different indices", () => {
		const partialCalls = [
			{
				index: 0,
				id: "call_1",
				function: { name: "func1", arguments: '{"a":' },
			},
			{
				index: 1,
				id: "call_2",
				function: { name: "func2", arguments: '{"b":' },
			},
			{
				index: 0,
				function: { arguments: '"value1"}' },
			},
			{
				index: 1,
				function: { arguments: '"value2"}' },
			},
		];

		const result = processPartialToolCalls(partialCalls);
		expect(result).toHaveLength(2);

		const call1 = result.find((call) => call.toolCallId === "call_1");
		const call2 = result.find((call) => call.toolCallId === "call_2");

		expect(call1?.args).toBe('{"a":"value1"}');
		expect(call2?.args).toBe('{"b":"value2"}');
	});
});

describe("processToolCalls", () => {
	it("should process OpenAI format tool calls", () => {
		const output = {
			tool_calls: [
				{
					id: "call_123",
					type: "function",
					function: {
						name: "test_function",
						arguments: '{"param": "value"}',
					},
				},
			],
		};

		const result = processToolCalls(output);
		expect(result).toEqual([
			{
				toolCallType: "function",
				toolCallId: "call_123",
				toolName: "test_function",
				args: '{"param": "value"}',
			},
		]);
	});

	it("should handle tool calls with object arguments", () => {
		const output = {
			tool_calls: [
				{
					id: "call_123",
					type: "function",
					function: {
						name: "test_function",
						arguments: { param: "value" },
					},
				},
			],
		};

		const result = processToolCalls(output);
		expect(result[0].args).toBe('{"param":"value"}');
	});

	it("should handle tool calls without function wrapper", () => {
		const output = {
			tool_calls: [
				{
					name: "test_function",
					arguments: '{"param": "value"}',
				},
			],
		};

		const result = processToolCalls(output);
		expect(result).toEqual([
			{
				toolCallType: "function",
				toolCallId: "test_function",
				toolName: "test_function",
				args: '{"param": "value"}',
			},
		]);
	});

	it("should return empty array when no tool calls present", () => {
		expect(processToolCalls({})).toEqual([]);
		expect(processToolCalls({ tool_calls: null })).toEqual([]);
		expect(processToolCalls({ tool_calls: [] })).toEqual([]);
	});

	it("should handle undefined or null arguments", () => {
		const output = {
			tool_calls: [
				{
					id: "call_123",
					function: {
						name: "test_function",
						arguments: null,
					},
				},
			],
		};

		const result = processToolCalls(output);
		expect(result[0].args).toBe("{}");
	});
});

describe("prepareToolsAndToolChoice", () => {
	it.todo("should send the right format");
});
