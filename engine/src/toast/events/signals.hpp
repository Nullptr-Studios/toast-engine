/**
 * @file signals.hpp
 * @author Dante Harper
 * @date 17 Jul 26
 */

#pragma once

#include "toast/world/box.hpp"

#include <functional>

namespace toast {
class Node;
}

namespace signals {
namespace _detail { }

template<typename F, typename... Args>
concept SignalCallback = std::is_invocable_r_v<void, F, Args...> ||    //
                         std::is_invocable_r_v<void, F>;               //

template<typename... Args>
class Signal {
	using callback_t = std::function<void(Args...)>;

	struct SigGroup {
		toast::Box<toast::Node> node;
		std::string identifier;
		callback_t cb;
	};

	struct {
		std::vector<SigGroup> listeners;
	} m;

public:
	auto listeners() -> std::vector<SigGroup>&;

	template<typename F>
	  requires SignalCallback<F, Args...>
	void subscribe(toast::Node& node, F&& cb);

	void subscribe(toast::Node& node, std::string_view identifier);

	void unsubscribe(toast::Node& node, std::string_view identifier);

	void clear() { m.listeners.clear(); }

	void fire(Args... args);
};

}
