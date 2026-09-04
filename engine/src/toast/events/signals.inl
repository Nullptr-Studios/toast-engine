#pragma once
#include "signals.hpp"
#include "toast/world/node.hpp"

namespace signals {

template<typename... Args>
template<typename F>
  requires SignalCallback<F, Args...>
inline void Signal<Args...>::subscribe(toast::Node& node, F&& cb) {
	callback_t wrapper = [f = std::forward<F>(cb)](Args... args) mutable {
		if constexpr (std::is_invocable_r_v<void, F, Args...>) {
			f(args...);
		} else if constexpr (std::is_invocable_r_v<void, F>) {
			f();
		}
	};
	m.listeners.push_back({
	  .node = toast::Box<toast::Node>(node),    //
	  .identifier = "Unnamed",
	  .cb = std::move(wrapper)                  //
	});
}

template<typename... Args>
inline void Signal<Args...>::subscribe(toast::Node& node, std::string_view identifier) {
	callback_t wrapper = [iden = std::string(identifier), box = toast::Box<toast::Node>(node)](Args... args) {
		box->call(iden, args...);
	};
	m.listeners.push_back({
	  .node = toast::Box<toast::Node>(node),    //
	  .identifier = std::string(identifier),
	  .cb = std::move(wrapper)                  //
	});
}

template<typename... Args>
inline void Signal<Args...>::unsubscribe(toast::Node& node, std::string_view identifier) {
	std::erase_if(m.listeners, [&](const SigGroup& listener) {
		return listener.node == toast::Box<toast::Node>(node) && listener.identifier == identifier;
	});
}

template<typename... Args>
inline void Signal<Args...>::fire(Args... args) {
	std::erase_if(m.listeners, [](const SigGroup& listener) {
		return not listener.node;    //
	});
	for (auto& listener : m.listeners) {
		if (listener.node.enabled()) {
			listener.cb(args...);
		}
	}
}

template<typename... Args>
inline auto Signal<Args...>::listeners() -> std::vector<SigGroup>& {
	return m.listeners;
}
}
