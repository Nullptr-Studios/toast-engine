/**
 * @file test_nodes.h
 * @author Xein
 * @date 06 Jul 2026
 *
 * @brief TODO: Brief description of the file's purpose
 */

#pragma once
#include <toast/log.hpp>
#include <toast/world/node_3d.hpp>

namespace toast {

class [[ToastNode]] TestSignal : Node {
public:
	signals::Signal<> void_signal;
	signals::Signal<int> int_signal;
	signals::Signal<float, float> float_signal;
};

class [[ToastNode]] TestNode : Node {
public:
	[[Reflect]]
	void emptyFunction() { }

	[[Reflect]]
	void floatFunction(float param) { }

	[[Reflect]]
	auto invalidFunction() -> bool {
		return false;
	}
};

}
