/**
 * @file signals.hpp
 * @author Dante Harper
 * @date 17 Jul 26
 */

#pragma once

#include "toast/uid.hpp"
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
		toast::UID uid;
		std::string identifier;

		toast::Box<toast::Node> node;
		callback_t cb;
	};

	struct {
		std::vector<SigGroup> listeners;
	} m;

public:
	template<typename F>
	  requires SignalCallback<F, Args...>
	void subscribe(toast::Node& node, F&& cb);

	void subscribe(toast::Node& node, std::string_view identifier);

	void unsubscribe(toast::Node& node, std::string_view identifier);

	void clear() { m.listeners.clear(); }

	void fire(Args... args);

	template<typename NodeType, auto MemberPtr>
	static auto get(void* signal) -> std::vector<std::pair<toast::UID, std::string>>;

	template<typename NodeType, auto MemberPtr>
	static void set(void* signal, const std::vector<std::pair<toast::UID, std::string>>& data);
};

}
#ifndef NODEFILE
#include <toast/events/signals.inl>
#endif
