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
	static auto get(void* signal) -> std::vector<std::pair<uint64_t, std::string>> {
		if (!signal) {
			return {};
		}

		auto* node = static_cast<NodeType*>(signal);
		// Access member via pointer-to-member syntax (node->*MemberPtr)
		const auto& target_signal = node->*MemberPtr;

		std::vector<std::pair<uint64_t, std::string>> result;
		result.reserve(target_signal.m.listeners.size());

		for (const auto& group : target_signal.m.listeners) {
			result.emplace_back(reinterpret_cast<uint64_t>(group.node.get()), group.identifier);
		}

		return result;
	}

	template<typename NodeType, auto MemberPtr>
	static void set(void* signal, const std::vector<std::pair<uint64_t, std::string>>& data) {
		if (!signal) {
			return;
		}

		auto* node = static_cast<NodeType*>(signal);
		auto& target_signal = node->*MemberPtr;

		target_signal.m.listeners.clear();
		target_signal.m.listeners.reserve(data.size());

		for (const auto& [node_ptr_val, identifier] : data) {
			using SigGroupType = typename std::decay_t<decltype(target_signal)>::SigGroup;

			SigGroupType group;
			group.node = reinterpret_cast<toast::Node*>(node_ptr_val);
			group.identifier = identifier;
			group.cb = nullptr;

			target_signal.m.listeners.push_back(std::move(group));
		}
	}
};

}
