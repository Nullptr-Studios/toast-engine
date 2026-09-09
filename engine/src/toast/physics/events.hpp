/**
 * @file events.hpp
 * @author Xein
 * @date 09 Sep 2026
 * @brief Physics events related mainly to the subscription of nodes
 */

#pragma once
#include <toast/events/event.hpp>

namespace event {

struct AddRigidbody : Event<AddRigidbody> {
	toast::Box<toast::Node> node;
};

struct RemoveRigidbody : Event<RemoveRigidbody> {
	toast::Box<toast::Node> node;
};

struct AddCollider : Event<AddCollider> {
	toast::Box<toast::Node> node;
};

struct RemoveCollider : Event<RemoveCollider> {
	toast::Box<toast::Node> node;
};

}
