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
template<typename NodeType, auto MemberPtr>
inline void Signal<Args...>::set(void* signal, const std::vector<std::pair<toast::UID, std::string>>& data) {
	if (!signal) {
		return;
	}
	auto* node = static_cast<NodeType*>(signal);
	auto& target_signal = node->*MemberPtr;
	using SigGroupType = typename std::decay_t<decltype(target_signal)>::SigGroup;

	target_signal.m.listeners.clear();
	target_signal.m.listeners.reserve(data.size());
	for (const auto& [uid, fn_identifier] : data) {
		toast::Box<toast::Node> box(node->find(uid));
		if (not box) {
			continue;
		}
		callback_t wrapper = [iden = std::string(fn_identifier), box](Args... args) {
			const_cast<toast::Box<toast::Node>&>(box)->call(iden, args...);    //
		};
		target_signal.m.listeners.push_back(
		    SigGroupType {
		      .uid = uid,
		      .identifier = fn_identifier,
		      .node = box,
		      .cb = std::move(wrapper),
		    }
		);
	}
}

template<typename... Args>
template<typename NodeType, auto MemberPtr>
inline auto Signal<Args...>::get(void* signal) -> std::vector<std::pair<toast::UID, std::string>> {
	if (!signal) {
		return {};
	}
	auto* node = static_cast<NodeType*>(signal);
	// Access member via pointer-to-member syntax (node->*MemberPtr)
	const auto& target_signal = node->*MemberPtr;
	std::vector<std::pair<toast::UID, std::string>> result;
	result.reserve(target_signal.m.listeners.size());
	for (const auto& group : target_signal.m.listeners) {
		result.emplace_back(group.uid, group.identifier);
	}
	return result;
}

}
